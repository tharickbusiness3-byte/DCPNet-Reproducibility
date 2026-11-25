from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, p, y):
        p = p.float()
        y = y.float()
        dims = tuple(range(2, p.ndim))
        inter = (p*y).sum(dims)
        den = p.sum(dims) + y.sum(dims)
        dice = (2*inter + self.eps) / (den + self.eps)
        dl = 1.0 - dice.mean()
        bce = F.binary_cross_entropy(p.clamp(self.eps, 1-self.eps), y)
        return dl + bce

def violation_vector(prob, brain):
    p = prob.float()
    wt,tc,et = p[:,0:1], p[:,1:2], p[:,2:3]
    v1 = F.relu(tc-wt).pow(2).mean()
    v2 = F.relu(et-tc).pow(2).mean()
    v3 = (wt*(1-brain.float())).pow(2).mean()
    return torch.stack([v1,v2,v3])

def hierarchy_violation(prob):
    p = prob.float()
    wt,tc,et = p[:,0:1], p[:,1:2], p[:,2:3]
    return F.relu(tc-wt).mean() + F.relu(et-tc).mean()

class DualController:
    def __init__(self, device, dual_lr=0.05, cap=5.0, tolerances=(1e-4,1e-4,1e-5)):
        self.lambdas = torch.ones(3, device=device)
        self.dual_lr = dual_lr
        self.cap = cap
        self.tolerances = torch.tensor(tolerances, device=device)

    @torch.no_grad()
    def update(self, v):
        self.lambdas.add_(self.dual_lr*(v.detach()-self.tolerances))
        self.lambdas.clamp_(0, self.cap)

    def state_dict(self):
        return {"lambdas": self.lambdas.detach().cpu(),
                "dual_lr": self.dual_lr, "cap": self.cap,
                "tolerances": self.tolerances.detach().cpu()}

    def load_state_dict(self, s):
        self.lambdas.copy_(s["lambdas"].to(self.lambdas.device))

def total_loss(outputs, target, segloss, dual, cfg):
    final = outputs["final"]
    raw = outputs["raw"]
    corrected = outputs["corrected"]
    brain = outputs["brain"]

    lseg = segloss(final, target)
    lraw = segloss(raw, target)
    vraw = violation_vector(raw, brain)
    vcorr = violation_vector(corrected, brain)
    lp = (dual.lambdas * vraw).sum()

    margin = float(cfg["loss"].get("feedback_margin", 0.001))
    lfb = F.relu(hierarchy_violation(corrected) - hierarchy_violation(raw) + margin)

    brier = (final.float() - target.float()).pow(2).mean()
    L = (
        lseg
        + cfg["loss"]["alpha_raw"]*lraw
        + cfg["loss"]["beta_predicate"]*lp
        + cfg["loss"]["gamma_feedback"]*lfb
        + cfg["loss"]["eta_brier"]*brier
    )
    return {
        "total": L, "seg": lseg, "raw_aux": lraw, "predicate": lp,
        "feedback": lfb, "brier": brier, "vraw": vraw, "vcorr": vcorr
    }
