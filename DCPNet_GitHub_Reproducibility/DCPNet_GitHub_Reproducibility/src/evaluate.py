from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from .metrics import dice_np, hd95_np, ece_binary, final_hierarchy_violations

@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    rows=[]
    roi=tuple(cfg["infer"]["roi_size"])
    thr=float(cfg["infer"]["threshold"])
    for b in loader:
        x=b["image"].to(device); y=b["label"].cpu().numpy()[0]
        if device.type=="cuda": torch.cuda.synchronize()
        t0=time.time()
        pred=sliding_window_inference(
            x,roi,cfg["infer"]["sw_batch_size"],
            lambda z:model(z)["final"], overlap=cfg["infer"]["overlap"]
        )
        if device.type=="cuda": torch.cuda.synchronize()
        latency=time.time()-t0
        prob=pred.float().cpu().numpy()[0]
        pb=prob>=thr
        case_id=b.get("case_id",["unknown"])[0]
        dices=[dice_np(pb[i],y[i]) for i in range(3)]
        hds=[hd95_np(pb[i],y[i]) for i in range(3)]
        ece=np.mean([ece_binary(prob[i],y[i]) for i in range(3)])
        brier=float(np.mean((prob-y)**2))
        rows.append({
            "case_id":case_id,
            "Dice_WT":dices[0],"Dice_TC":dices[1],"Dice_ET":dices[2],
            "macro_Dice":float(np.mean(dices)),
            "HD95_WT":hds[0],"HD95_TC":hds[1],"HD95_ET":hds[2],
            "ECE":float(ece),"Brier":brier,
            "final_hierarchy_violation_voxels":final_hierarchy_violations(pred),
            "latency_s":latency,
        })
    return pd.DataFrame(rows)
