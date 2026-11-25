from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice_np(pred, gt, eps=1e-8):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if not pred.any() and not gt.any(): return 1.0
    if not pred.any() or not gt.any(): return 0.0
    return (2*(pred&gt).sum()+eps)/(pred.sum()+gt.sum()+eps)

def hd95_np(pred, gt, spacing=(1,1,1)):
    pred = pred.astype(bool); gt = gt.astype(bool)
    if not pred.any() and not gt.any(): return np.nan
    if not pred.any() or not gt.any(): return np.nan
    ps = pred ^ binary_erosion(pred)
    gs = gt ^ binary_erosion(gt)
    d_gt = distance_transform_edt(~gs, sampling=spacing)
    d_pr = distance_transform_edt(~ps, sampling=spacing)
    d = np.concatenate([d_gt[ps], d_pr[gs]])
    return float(np.percentile(d,95))

def ece_binary(prob, target, bins=15):
    p = prob.reshape(-1)
    y = target.reshape(-1)
    edges = np.linspace(0,1,bins+1)
    out = 0.0
    n = len(p)
    for i in range(bins):
        m = (p>=edges[i]) & (p < edges[i+1] if i<bins-1 else p<=edges[i+1])
        if m.any():
            conf = p[m].mean()
            acc = y[m].mean()
            out += m.sum()/n * abs(conf-acc)
    return float(out)

def final_hierarchy_violations(prob, tol=1e-6):
    p = prob.float()
    wt,tc,et = p[:,0:1],p[:,1:2],p[:,2:3]
    return int(((tc > wt+tol) | (et > tc+tol)).sum().item())
