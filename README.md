# Range-gated zero-shot denoising

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

Use `--dataset mayo-d45` for D45. Use `--epochs 500` for the full setting.

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
  --epochs 500 \
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
  --epochs 500 \
  --out-dir outputs/fmd
```
