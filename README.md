# Range-Gated Zero-Shot Denoising

This repository contains code for a zero-shot biomedical image denoising method based on range-gated adaptive kernel prediction.

The method trains an image-specific model from a single noisy image. It does not require external clean/noisy image pairs, pretrained weights, or supervised clean targets. The model predicts local adaptive kernels, applies an intensity-based range gate to preserve structures, and optimizes using self-supervised losses.

## Repository contents

```text
scripts/
  denoise_one_image_rangegated_hpc.py        Main range-gated adaptive KPN implementation
  run_npy_rangegated_ours_f2n.py             CT benchmark on prepared .npy pairs

  mayo/
    make_dicom_manifest_rangegated.py        Build Mayo low/full-dose DICOM manifest
    run_rangegated_dicom_manifest_batch.py   Run Mayo DICOM benchmark from manifest
    summarize_rangegated_dicom_results.py    Summarize Mayo result CSVs

  fmd/
    zero_shot_denoising_core.py              Shared zero-shot utilities and models
    fmd_paired_zero_shot_benchmark.py        FMD benchmark: noisy, ZS-N2N, ours
    fmd_ours_only_categories.py              FMD evaluation for our method only
```

## Installation

```bash
conda create -n rangegated python=3.10 -y
conda activate rangegated
pip install -r requirements.txt
```

For GPU use, install the PyTorch build that matches your CUDA version.


## Run one CT image

```bash
python scripts/denoise_one_image_rangegated_hpc.py \
  --low /path/to/low.png \
  --full /path/to/full.png \
  --out-dir outputs/one_image_test
```

## Run Mayo DICOM benchmark

Create a manifest:

```bash
python scripts/mayo/make_dicom_manifest_rangegated.py \
  --b30-root "1mm B30" \
  --d45-root "1mm D45" \
  --patients L096 L109 L506 \
  --kernels B30 D45 \
  --out manifests/dicom_rangegated_manifest.csv
```

Run a batch:

```bash
python scripts/mayo/run_rangegated_dicom_manifest_batch.py \
  --manifest manifests/dicom_rangegated_manifest.csv \
  --batch-index 0 \
  --batch-size 50 \
  --out-csv results/mayo_batch_000.csv
```

Summarize batches:

```bash
python scripts/mayo/summarize_rangegated_dicom_results.py \
  "results/mayo_batch_*.csv" \
  --out-dir results/mayo_summary
```

## Run FMD paired benchmark

Expected FMD structure:

```test
FMD_ROOT/
  raw/
    Confocal_BPAE_B_1.png
    TwoPhoton_BPAE_B_2.png
  gt/
    Confocal_BPAE_B_1.png
    TwoPhoton_BPAE_B_2.png
```

Run the paired benchmark:

```bash
python scripts/fmd/fmd_paired_zero_shot_benchmark.py \
  --root /path/to/FMD_ROOT \
  --channels gray \
  --crop 256 \
  --out-dir results/fmd_paired \
  --save-images
```

Run only our method by category:

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

## Baselines

The FMD paired benchmark includes a ZS-N2N baseline implementation in `scripts/fmd/zero_shot_denoising_core.py`.

The CT scripts optionally call F2N if the F2N implementation is available locally. This repository does not redistribute third-party F2N code. Place the F2N package or `filter2noise.py` according to the import instructions in `scripts/denoise_one_image_rangegated_hpc.py`.

## Citation

Citation information can be added after the paper is finalized.
