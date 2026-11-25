from __future__ import annotations
import time, csv
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .model import DCPNet
from .losses import DiceBCELoss, DualController, total_loss
from .utils import atomic_torch_save, capture_rng_state, restore_rng_state

def _checkpoint(model,opt,sched,scaler,dual,epoch,best,bad,history,cfg,split_hash,overflow_count):
    return {
        "epoch":epoch, "model":model.state_dict(), "optimizer":opt.state_dict(),
        "scheduler":sched.state_dict(), "scaler":scaler.state_dict(),
        "dual":dual.state_dict(), "best_macro":best, "bad_epochs":bad,
        "history":history, "config":cfg, "split_sha256":split_hash,
        "overflow_count":overflow_count, "rng":capture_rng_state(),
    }

def train_model(loaders, cfg, out_dir, split_hash, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCPNet(tuple(cfg["model"]["channels"]),
                   cfg["model"]["refinement_fp32"],
                   cfg["model"]["decoder2_fp32"]).to(device)
    segloss = DiceBCELoss()
    dual = DualController(device, cfg["loss"]["dual_lr"], cfg["loss"]["dual_cap"],
                          tuple(cfg["loss"]["tolerances"]))
    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=cfg["train"]["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type=="cuda")

    start_epoch,best,bad,history,overflow_count = 1,-1.0,0,[],0
    if resume:
        ck = torch.load(resume, map_location=device)
        if ck["split_sha256"] != split_hash:
            raise RuntimeError("Split hash mismatch")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        dual.load_state_dict(ck["dual"])
        restore_rng_state(ck.get("rng"))
        start_epoch = ck["epoch"]+1
        best = ck["best_macro"]; bad = ck["bad_epochs"]; history = ck["history"]
        overflow_count = ck.get("overflow_count",0)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg["train"]["max_epochs"]+1):
        model.train()
        sums = {"total":0.0,"n":0}
        t0=time.time()

        for batch in loaders["train"]:
            x=batch["image"].to(device); y=batch["label"].to(device)
            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                out=model(x)
                losses=total_loss(out,y,segloss,dual,cfg)
                loss=losses["total"]

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite forward loss")

            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip"])
            scaler.step(opt)
            scaler.update()
            new_scale = scaler.get_scale()
            overflow = new_scale < old_scale

            if overflow:
                overflow_count += 1
                opt.zero_grad(set_to_none=True)
                # important: skip dual update on overflowed optimizer step
                continue
            dual.update(losses["vraw"])
            sums["total"] += float(loss.detach())
            sums["n"] += 1

        sched.step()
        # Validation should be implemented with sliding-window inference;
        # caller may replace this hook with full evaluate_loader().
        val_macro = validate_macro_dice(model, loaders["val"], cfg, device)

        mean_loss=sums["total"]/max(sums["n"],1)
        row={"epoch":epoch,"train_loss":mean_loss,"val_macro_dice":val_macro,
             "lr":opt.param_groups[0]["lr"],"overflow_count":overflow_count,
             "duration_s":time.time()-t0}
        history.append(row)

        improved=val_macro > best+1e-4
        if improved:
            best=val_macro; bad=0
        else:
            bad+=1

        ck=_checkpoint(model,opt,sched,scaler,dual,epoch,best,bad,history,cfg,split_hash,overflow_count)
        atomic_torch_save(ck,out_dir/"last.pt")
        if improved:
            atomic_torch_save(ck,out_dir/"best.pt")

        if bad >= cfg["train"]["early_stopping_patience"]:
            break

    return model, history

@torch.no_grad()
def validate_macro_dice(model, loader, cfg, device):
    from monai.inferers import sliding_window_inference
    vals=[]
    model.eval()
    roi=tuple(cfg["infer"]["roi_size"])
    thr=cfg["infer"]["threshold"]
    for b in loader:
        x=b["image"].to(device); y=b["label"].to(device)
        pred = sliding_window_inference(
            x, roi, cfg["infer"]["sw_batch_size"],
            lambda z: model(z)["final"], overlap=cfg["infer"]["overlap"]
        )
        pb=(pred>=thr).float()
        dims=tuple(range(2,pb.ndim))
        inter=(pb*y).sum(dims); den=pb.sum(dims)+y.sum(dims)
        d=(2*inter+1e-6)/(den+1e-6)
        vals.append(float(d.mean()))
    return float(sum(vals)/max(len(vals),1))
