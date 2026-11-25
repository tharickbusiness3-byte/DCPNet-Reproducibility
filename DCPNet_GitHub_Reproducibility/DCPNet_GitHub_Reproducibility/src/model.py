from __future__ import annotations
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

def _autocast_off(x):
    if x.is_cuda:
        return torch.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(cout, affine=True)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(cout, affine=True)
        self.act = nn.GELU()
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv3d(cin, cout, 1, stride=stride, bias=False)

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)

class UpBlock(nn.Module):
    def __init__(self, cin, skip_c, cout):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        self.res = ResBlock(cout + skip_c, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))

class DCPNet(nn.Module):
    def __init__(self, channels=(24,48,96,192), refinement_fp32=True, decoder2_fp32=True):
        super().__init__()
        c0,c1,c2,c3 = channels
        self.refinement_fp32 = refinement_fp32
        self.decoder2_fp32 = decoder2_fp32

        self.stems = nn.ModuleList([ResBlock(1, c0) for _ in range(4)])
        self.gate_mlp = nn.Sequential(
            nn.Linear(4*c0, c0),
            nn.GELU(),
            nn.Linear(c0, 4)
        )

        self.enc1 = ResBlock(c0, c1, stride=2)
        self.enc2 = ResBlock(c1, c2, stride=2)
        self.enc3 = ResBlock(c2, c3, stride=2)

        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)
        self.dec0 = UpBlock(c1, c0, c0)

        self.raw_head = nn.Conv3d(c0, 3, 1)
        # decoder feature + r1+r2+r3+uncertainty = c0+4
        self.refinement = nn.Sequential(
            ResBlock(c0+4, c0),
            ResBlock(c0, c0),
            nn.Conv3d(c0, 3, 1)
        )

    @staticmethod
    def brain_mask(x):
        return (x.abs().sum(dim=1, keepdim=True) > 0).float()

    @staticmethod
    def entropy(p, eps=1e-6):
        p = p.float().clamp(eps, 1-eps)
        h = -(p*torch.log(p) + (1-p)*torch.log(1-p))
        return h.mean(dim=1, keepdim=True)

    def forward(self, x):
        availability = (x.abs().flatten(2).sum(dim=-1) > 0).float()  # [B,4]
        feats = [stem(x[:,i:i+1]) for i, stem in enumerate(self.stems)]
        pooled = torch.cat([f.mean(dim=(2,3,4)) for f in feats], dim=1)
        logits = self.gate_mlp(pooled)
        logits = logits.masked_fill(availability <= 0, -1e9)
        gates = torch.softmax(logits, dim=1)
        fused = sum(feats[i] * gates[:,i].view(-1,1,1,1,1) for i in range(4))

        e1 = self.enc1(fused)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        if self.decoder2_fp32:
            with _autocast_off(e3):
                d2 = self.dec2(e3.float(), e2.float())
        else:
            d2 = self.dec2(e3, e2)

        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, fused)

        raw_logits = self.raw_head(d0)
        raw = torch.sigmoid(raw_logits.float())
        wt, tc, et = raw[:,0:1], raw[:,1:2], raw[:,2:3]
        r1 = F.relu(tc - wt)
        r2 = F.relu(et - tc)
        brain = self.brain_mask(x)
        r3 = wt * (1.0 - brain)
        unc = self.entropy(raw)

        rin = torch.cat([d0.float(), r1, r2, r3, unc], dim=1)
        if self.refinement_fp32:
            with _autocast_off(rin):
                delta = self.refinement.float()(rin.float())
        else:
            delta = self.refinement(rin)

        corrected_logits = raw_logits.float() + delta.float()
        corrected = torch.sigmoid(corrected_logits)

        cw, ct, ce = corrected[:,0:1], corrected[:,1:2], corrected[:,2:3]
        final_w = cw
        final_t = final_w * ct
        final_e = final_t * ce
        final = torch.cat([final_w, final_t, final_e], dim=1)

        return {
            "raw_logits": raw_logits,
            "corrected_logits": corrected_logits,
            "raw": raw,
            "corrected": corrected,
            "final": final,
            "violation_maps": (r1, r2, r3),
            "gates": gates,
            "brain": brain,
        }
