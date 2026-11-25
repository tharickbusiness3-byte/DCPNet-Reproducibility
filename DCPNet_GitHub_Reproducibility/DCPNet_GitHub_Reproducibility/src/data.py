from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd,
    Spacingd, NormalizeIntensityd, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandScaleIntensityd, RandShiftIntensityd, MapTransform
)

MODALITY_SUFFIXES = ("t1n", "t1c", "t2w", "t2f")

class BraTSToRegionsd(MapTransform):
    def __init__(self, keys=("label",)):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        seg = d["label"]
        if not torch.is_tensor(seg):
            seg = torch.as_tensor(seg)
        # expected [1,D,H,W]
        s = seg[0] if seg.ndim == 4 else seg
        uniq = torch.unique(s)
        et_label = 4 if torch.any(uniq == 4) else 3
        wt = (s > 0)
        tc = (s == 1) | (s == et_label)
        et = (s == et_label)
        y = torch.stack([wt, tc, et], dim=0).float()
        assert torch.all(y[2] <= y[1]) and torch.all(y[1] <= y[0])
        d["label"] = y
        return d

def case_record(data_root: str | Path, case_id: str) -> Dict:
    c = Path(data_root) / case_id
    images = []
    for suffix in MODALITY_SUFFIXES:
        m = list(c.glob(f"*{suffix}.nii.gz"))
        if len(m) != 1:
            raise FileNotFoundError(f"{case_id}: expected one *{suffix}.nii.gz, found {len(m)}")
        images.append(str(m[0]))
    segs = list(c.glob("*seg.nii.gz"))
    if len(segs) != 1:
        raise FileNotFoundError(f"{case_id}: expected one *seg.nii.gz, found {len(segs)}")
    return {"case_id": case_id, "image": images, "label": str(segs[0])}

def load_split_records(data_root: str | Path, split_json: str | Path):
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    return {
        k: [case_record(data_root, cid) for cid in split[k]]
        for k in ("train", "val", "test")
    }

def build_transforms(patch_size=(96,96,96), spacing=(1,1,1), train=True):
    base = [
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        EnsureTyped(keys=["image","label"], track_meta=True),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        Spacingd(keys=["image","label"], pixdim=spacing, mode=("bilinear","nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image","label"], source_key="image"),
        BraTSToRegionsd(keys=["label"]),
    ]
    if train:
        base += [
            RandCropByPosNegLabeld(
                keys=["image","label"], label_key="label",
                spatial_size=patch_size, pos=2, neg=1, num_samples=2,
                image_key="image", image_threshold=0
            ),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.10, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.5),
        ]
    return Compose(base)

def build_loaders(records, cfg):
    patch = tuple(cfg["data"]["patch_size"])
    spacing = tuple(cfg["data"]["spacing"])
    train_ds = Dataset(records["train"], build_transforms(patch, spacing, train=True))
    val_ds = Dataset(records["val"], build_transforms(patch, spacing, train=False))
    test_ds = Dataset(records["test"], build_transforms(patch, spacing, train=False))
    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])
    return {
        "train": DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=nw, pin_memory=True),
    }
