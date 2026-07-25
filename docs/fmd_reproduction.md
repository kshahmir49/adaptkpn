# FMD microscopy reproduction

Expected folder structure:

```text
FMD_ROOT/
  raw/
    Confocal_BPAE_B_1.png
    TwoPhoton_BPAE_B_2.png
  gt/
    Confocal_BPAE_B_1.png
    TwoPhoton_BPAE_B_2.png
```

## Paired benchmark

```bash
python scripts/fmd/fmd_paired_zero_shot_benchmark.py \
  --root /path/to/FMD_ROOT \
  --channels gray \
  --crop 256 \
  --out-dir results/fmd_paired \
  --save-images
```

This compares the noisy input, ZS-N2N, and the proposed method.

## Ours only by category

```bash
python scripts/fmd/fmd_ours_only_categories.py \
  --root /path/to/FMD_ROOT \
  --categories TwoPhoton_MICE TwoPhoton_BPAE Confocal_BPAE \
  --channels gray \
  --crop 256 \
  --range-sigma-init 0.08 \
  --lambda-edge 350 \
  --out-dir results/fmd_ours_only
```
