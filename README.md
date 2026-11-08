# AdaptKPN

AdaptKPN is a zero-shot image denoising method based on learned spatial kernel prediction with an intensity range gate. Each image is optimized independently and no clean target is used during training.

This repository contains the final implementation used for the CT and fluorescence microscopy experiments.

## Final model configuration

| Setting | Value |
| --- | --- |
| Stages | 3 |
| Spatial kernel | 5 x 5 |
| Hidden channels | 16 |
| Stage output | 0.75 filtered + 0.25 input |
| Range sigma initialization | 0.08 |
| Range sigma bounds | 0.005 to 0.25 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Scheduler | OneCycleLR |
| Edge-loss weight | 350 |
| EMA | 0.999 |
| AMP | enabled on CUDA |
| Self ensemble | disabled |
| Poisson auxiliary loss | disabled |

The grayscale model has 2,649 trainable parameters.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For GPU use, install a PyTorch build appropriate for the local CUDA environment before `pip install -e .` if needed.

## Repository layout

```text
adaptkpn/
  core.py                     canonical final AdaptKPN implementation
baselines/
  zsn2n.py                    ZS-N2N reproduction used for the FMD comparison
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
  runtime/
    benchmark_ct_f2n.py
slurm/
  ct_array.sh
  fmd_array.sh
benchmarks/
  fmd_published_psnr.csv
```

## CT experiments

CT input is paired quarter-dose and full-dose DICOM or IMA data. The clean image is used only for PSNR and SSIM evaluation.

Create one manifest for each patient and reconstruction kernel.

```bash
python scripts/ct/make_manifest.py \
  --low-dir /path/to/quarter-dose \
  --full-dir /path/to/full-dose \
  --patient PATIENT_ID \
  --kernel D45 \
  --out manifests/d45_patient.csv
```

The manifest builder first pairs slices by `ImagePositionPatient[2]` and falls back to `InstanceNumber`.

### D45

The final D45 protocol uses the raw DICOM pixel array, converted to float32 and scaled as

```text
clip(pixel_array / 4095, 0, 1)
```

Run one slice locally with

```bash
python scripts/ct/run.py \
  --manifest manifests/d45_patient.csv \
  --normalization fixed4095 \
  --batch-index 0 \
  --batch-size 1 \
  --out-dir outputs/d45
```

D45 uses 500 optimization epochs.

### B30

The final B30 protocol uses independent per-slice min-max normalization to `[0, 1]` for both noisy and full-dose images.

```bash
python scripts/ct/run.py \
  --manifest manifests/b30_patient.csv \
  --normalization minmax \
  --batch-index 0 \
  --batch-size 1 \
  --out-dir outputs/b30
```

B30 also uses 500 optimization epochs.

### CT summary

```bash
python scripts/ct/summarize.py "outputs/d45/ct_batch_*.csv" --out d45_summary.csv
```

## FMD fluorescence microscopy experiment

Expected dataset layout

```text
FMD/
  raw/
    Confocal_BPAE_*.png
    TwoPhoton_BPAE_*.png
    TwoPhoton_MICE_*.png
  gt/
    matching filenames
```

The final FMD protocol uses full-resolution grayscale images. Images are converted to grayscale uint8, divided by 255, and only cropped by one row or column if required to make the spatial dimensions even. There is no 256 x 256 center crop.

Create the manifest

```bash
python scripts/fmd/make_manifest.py \
  --root /path/to/FMD \
  --out manifests/fmd.csv
```

Run AdaptKPN for the final 2,000-step FMD protocol

```bash
python scripts/fmd/run.py \
  --manifest manifests/fmd.csv \
  --batch-index 0 \
  --batch-size 1 \
  --epochs 2000 \
  --out-dir outputs/fmd \
  --save-images
```

To reproduce the same-image ZS-N2N comparison, add `--run-zsn2n`.

```bash
python scripts/fmd/run.py \
  --manifest manifests/fmd.csv \
  --batch-index 0 \
  --batch-size 1 \
  --epochs 2000 \
  --run-zsn2n \
  --out-dir outputs/fmd \
  --save-images
```

Summarize all completed jobs

```bash
python scripts/fmd/summarize.py "outputs/fmd/fmd_batch_*.csv" --out-prefix fmd_final
```

The summary reports per-category means and the simple mean across the three benchmark categories.

## Slurm

The scripts in `slurm/` intentionally contain no account names, project identifiers, environment names, usernames, institutional paths, or other cluster-specific details.

Customize only the commented placeholders for your cluster and environment.

```bash
# Example CT array submission
BATCH_SIZE=1 MAX_PARALLEL=8 OUT_DIR=outputs/d45 \
  bash scripts/ct/submit_slurm.sh manifests/d45_patient.csv fixed4095

# Example FMD array submission
BATCH_SIZE=1 MAX_PARALLEL=8 OUT_DIR=outputs/fmd \
  bash scripts/fmd/submit_slurm.sh manifests/fmd.csv
```

## Runtime comparison with F2N

The repository does not vendor the F2N implementation. Clone the official Filter2Noise repository separately and pass its `filter2noise.py` path.

The timing script measures fresh model construction, per-image optimization, and the final denoising forward pass for both methods. Image loading and metric computation are excluded. A warm-up image is discarded and the execution order alternates across measured slices.

```bash
python scripts/runtime/benchmark_ct_f2n.py \
  --manifest manifests/d45_timing.csv \
  --f2n-path /path/to/Filter2Noise/filter2noise.py \
  --warmup 1 \
  --n 30 \
  --paper-f2n-seconds 16 \
  --out ct_runtime_results.csv
```

The hardware-normalized runtime printed by this script is an estimate. The measured same-GPU times remain the primary reproducible timing result.

## Reproducibility notes

- Seed 123 is used by default.
- Ground truth is never passed to the optimization routine.
- CT uses 500 epochs.
- FMD uses 2,000 epochs for the final benchmark protocol.
- No test-time self-ensemble is used.
- The same canonical implementation in `adaptkpn/core.py` is used by both CT and FMD runners.
- The FMD ZS-N2N implementation is provided only to reproduce the controlled same-image comparison.

## References

- Mansour and Heckel, *Zero-Shot Noise2Noise: Efficient Image Denoising without any Data*, CVPR 2023.
- Filter2Noise is an external baseline and is not redistributed in this repository.
