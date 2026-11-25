# DCP-Net: Differentiable Clinical Predicates for Hierarchy-Consistent Brain Tumor Segmentation

This repository contains a clean, reproducible implementation of DCP-Net for BraTS 2023 Adult Glioma MRI segmentation.

## Scientific scope

DCP-Net predicts three nested tumor regions:

- WT — Whole Tumor
- TC — Tumor Core
- ET — Enhancing Tumor

with the required hierarchy:

`ET ⊆ TC ⊆ WT`

The model combines:

1. modality-specific 3D stems,
2. learned modality reliability gates,
3. a compact residual 3D encoder-decoder,
4. raw WT/TC/ET proposal logits,
5. differentiable hierarchy-violation maps,
6. logic-to-feature refinement,
7. monotone hierarchical projection.

## Important reproducibility note

The training code supports an **80-epoch maximum schedule** (`max_epochs: 80`) with early stopping and checkpoint resume.

Do not describe a run as “80 epochs completed” unless its logs/checkpoints actually show 80 completed epochs.  
The training ceiling is a configuration, not evidence that every run reached that ceiling.

## Expected dataset

BraTS 2023 GLI TrainingData, with per-case files such as:

- `*-t1n.nii.gz`
- `*-t1c.nii.gz`
- `*-t2w.nii.gz`
- `*-t2f.nii.gz`
- `*-seg.nii.gz`

Native labels may be `{0,1,2,3}` or `{0,1,2,4}`.

Targets are converted to:

- WT = label > 0
- TC = label 1 OR enhancing label
- ET = enhancing label

## Colab setup

```bash
pip install -r requirements.txt
```

Example:

```bash
python run_full_pipeline.py \
  --data_root "/content/DCP_NET_PROJECT/data/brats2023_train/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" \
  --split_json "/content/DCP_NET_PROJECT/splits/split_seed17.json" \
  --output_dir "/content/DCP_NET_PROJECT/github_run"
```

## Main configuration

See `configs/dcpnet.yaml`.

Key defaults:

- seed: 17
- patch size: 96×96×96
- channels: 24, 48, 96, 192
- AdamW
- learning rate: 2e-4
- weight decay: 1e-5
- max epochs: 80
- early stopping patience: 15
- AMP enabled
- gradient clipping: 12
- refinement head executed in FP32 for numerical stability
- decoder level 2 can be executed in FP32 when enabled
- recoverable AMP overflow skips optimizer/dual update and continues
- final monotone projection guarantees ET ≤ TC ≤ WT

## Evaluation

`evaluate.py` computes:

- Dice WT / TC / ET
- Macro Dice
- HD95
- ECE
- Brier score
- raw/corrected/final hierarchy violations
- inference latency

Use a frozen checkpoint and frozen split. Do not tune on the locked test set.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── run_full_pipeline.py
├── configs/
│   └── dcpnet.yaml
├── src/
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── utils.py
└── scripts/
    └── colab_setup.py
```

## Citation

Add the final paper citation here after publication.
