# AdaptKPN

AdaptKPN is a zero-shot image denoising method based on learned spatial kernel prediction with an intensity range gate. Each image is optimized independently and no clean target is used during training.

This repository contains the final AdaptKPN implementation used for the CT and fluorescence microscopy experiments.

## Model configuration

| Setting | Value |
| --- | --- |
| Stages | 2 |
| Spatial kernel | 5 x 5 |
| Hidden channels | 16 |
| Stage output | 0.75 filtered + 0.25 input |
| Range sigma initialization | 0.08 |
| Range sigma bounds | 0.005 to 0.25 |
| Optimizer | AdamW |
| Learning rate | 1.6e-2 |
| Weight decay | 0.01 |
| Scheduler | OneCycleLR |
| Edge-loss weight | 350 (75 for CT B30) |
| EMA | off |
| AMP | enabled on CUDA |

The grayscale model has 1,766 trainable parameters.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Repository layout

```text
adaptkpn/
  core.py

scripts/
  ct/
    make_manifest.py
    run.py
    summarize.py
    submit_slurm.sh
  fmd/
    make_manifest.py
    run.py
    summarize.py
    submit_slurm.sh

slurm/
  ct_array.sh
  fmd_array.sh

tests/
  smoke_test.py
```

## CT experiments

CT input is paired quarter-dose and full-dose DICOM or IMA data. The reference image is used only for PSNR and SSIM evaluation.

Create a manifest

```bash
python scripts/ct/make_manifest.py \
  --low-dir /path/to/quarter-dose \
  --full-dir /path/to/full-dose \
  --patient PATIENT_ID \
  --kernel D45 \
  --out manifests/d45_patient.csv
```

### D45

D45 uses fixed raw scaling

```text
clip(pixel_array / 4095, 0, 1)
```

Run

```bash
python scripts/ct/run.py \
  --manifest manifests/d45_patient.csv \
  --normalization fixed4095 \
  --batch-index 0 \
  --batch-size 1 \
  --out-dir outputs/d45
```

### B30

B30 uses independent per-slice min-max normalization to `[0, 1]`.

```bash
python scripts/ct/run.py \
  --manifest manifests/b30_patient.csv \
  --normalization minmax \
  --lambda-edge 75 \
  --batch-index 0 \
  --batch-size 1 \
  --out-dir outputs/b30
```

CT experiments use 500 optimization epochs.

Summarize completed jobs

```bash
python scripts/ct/summarize.py "outputs/d45/ct_batch_*.csv" --out d45_summary.csv
```

## FMD fluorescence microscopy experiments

Expected dataset layout

```text
FMD/
  raw/
    ...
  gt/
    ...
```

The FMD protocol uses full-resolution grayscale images. Images are converted to grayscale, scaled to `[0, 1]`, and cropped only if needed to make the spatial dimensions even.

Create a manifest

```bash
python scripts/fmd/make_manifest.py \
  --root /path/to/FMD \
  --out manifests/fmd.csv
```

Run AdaptKPN

```bash
python scripts/fmd/run.py \
  --manifest manifests/fmd.csv \
  --batch-index 0 \
  --batch-size 1 \
  --epochs 2000 \
  --out-dir outputs/fmd \
  --save-images
```

Summarize completed jobs

```bash
python scripts/fmd/summarize.py "outputs/fmd/fmd_batch_*.csv" --out-prefix fmd_final
```

## Slurm

The example Slurm scripts use generic placeholders only. Customize them for your own cluster before submission.

```bash
BATCH_SIZE=1 MAX_PARALLEL=8 OUT_DIR=outputs/b30 LAMBDA_EDGE=75 \
  bash scripts/ct/submit_slurm.sh manifests/b30_patient.csv minmax

BATCH_SIZE=1 MAX_PARALLEL=8 OUT_DIR=outputs/d45 \
  bash scripts/ct/submit_slurm.sh manifests/d45_patient.csv fixed4095

BATCH_SIZE=1 MAX_PARALLEL=8 OUT_DIR=outputs/fmd \
  bash scripts/fmd/submit_slurm.sh manifests/fmd.csv
```

## Reproducibility

- Seed 123 is used by default.
- Ground truth is never passed to the optimization routine.
- CT uses 500 epochs.
- FMD uses 2,000 epochs.
- The same implementation in `adaptkpn/core.py` is used by both CT and FMD runners.
