# Reproducibility and reporting rules

1. Keep train/validation/test subject IDs frozen before final evaluation.
2. Do not tune threshold or architecture on the locked test set.
3. Report the actual completed epoch count from logs/checkpoints.
4. `max_epochs=80` means training was *allowed* to run up to 80 epochs; it does not mean 80 epochs were completed.
5. Report the selected best checkpoint and selection criterion.
6. Preserve undefined HD95 values as undefined rather than replacing them with arbitrary constants.
7. Report DCP-Net's consistency benefit separately from Dice accuracy.
8. If a baseline has higher Dice, state that explicitly.
