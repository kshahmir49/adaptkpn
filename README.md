# AdaptKPN, zero-shot denoising

AdaptKPN is a three-stage cascade of adaptive kernel-prediction stages (about 2.7k parameters).
Each stage predicts a 5x5 softmax kernel and a per-pixel blend alpha, and outputs
`alpha * weighted_average + (1 - alpha) * input`. It is trained per image with the zero-shot
Noise2Noise pair-downsampling loss plus an auxiliary Poisson-thinning loss, using Adam with StepLR,
EMA weights and an 8-fold rotation/flip self-ensemble at inference.

Notes on settings

- `--fixed-blend 0.8` uses the original constant 0.8/0.2 blend (the setup behind the first B30 numbers).
  Leave it unset for the learned per-pixel blend.
- `--blend-init` sets the initial blend logit. Natural-image runs used -3.0, -1.5 and -0.5 for sigma 10, 25 and 50. D45 used -1.0.
- DICOM input (`.dcm`, `.ima`) is converted to HU and clipped to `--window-min/--window-max` (default -3000 to 3000), then scaled to [0, 1].
  PNG input is scaled by its dtype range. Window choice changes absolute PSNR, so use the same window for every method you compare.
- Defaults are 2000 epochs, lr 1e-3, StepLR step 1000 gamma 0.5, Poisson peak 50 weight 0.25.
- This repo runs AdaptKPN only. Filter2Noise comparisons are run separately.

Files:

```text
denoise.py              train and run our method on one image or a manifest
make_mayo_manifest.py   create Mayo B30/D45 low/full DICOM manifests
make_fmd_manifest.py    create FMD raw/GT manifests
requirements.txt
```

## Install

```bash
conda create -n rangegated python=3.10 -y
conda activate rangegated
pip install -r requirements.txt
```

## One Mayo image

```bash
python denoise.py \
  --dataset mayo-b30 \
  --noisy /path/to/quarter.IMA \
  --target /path/to/full.IMA \
  --epochs 50 \
  --out-dir outputs/mayo_one
```

Use `--dataset mayo-d45` for D45. Use the default `--epochs 2000` for the full setting.

## Full Mayo folder

```bash
python make_mayo_manifest.py \
  --dataset mayo-b30 \
  --patient L096 \
  --low-dir "/path/to/quarter_1mm/L096/quarter_1mm" \
  --full-dir "/path/to/full_1mm/L096/full_1mm" \
  --out manifests/mayo_b30_l096.csv
```

```bash
python denoise.py \
  --dataset mayo-b30 \
  --manifest manifests/mayo_b30_l096.csv \
  --epochs 2000 \
  --out-dir outputs/mayo_b30_l096
```

## One FMD image

```bash
python denoise.py \
  --dataset fmd \
  --noisy /path/to/FMD/raw/TwoPhoton_BPAE_B_2.png \
  --target /path/to/FMD/gt/TwoPhoton_BPAE_B_2.png \
  --channels gray \
  --epochs 50 \
  --out-dir outputs/fmd_one
```

## Full FMD folder

```bash
python make_fmd_manifest.py \
  --root /path/to/FMD \
  --out manifests/fmd.csv \
  --channels gray
```

```bash
python denoise.py \
  --dataset fmd \
  --manifest manifests/fmd.csv \
  --epochs 2000 \
  --out-dir outputs/fmd
```
