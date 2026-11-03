# AdaptKPN

Zero-shot CT denoising. It trains on the single noisy image, so no clean data and no pretraining are needed.

## Install

```bash
pip install -r requirements.txt
```

## Run on one slice

```bash
python denoise.py --noisy low_dose.png --target full_dose.png
```

The denoised image and `results.csv` (PSNR and SSIM) are saved in `outputs/`.
`--target` is optional and is only used to report the scores.
For a quick test add `--epochs 200`.

## Run on a whole dataset

The PNG slices should be laid out like this.

```text
Preprocessed_512x512/512/
  Quarter Dose/1mm/Soft Kernel (B30)/L096/*.png
  Full Dose/1mm/Soft Kernel (B30)/L096/*.png
```

1. Make the list of slice pairs.

```bash
python make_png_manifest.py --root Preprocessed_512x512/512 \
  --kernel "Soft Kernel (B30)" --patients L096 L506 L109 --out manifests/b30.csv
```

2. Denoise them all.

```bash
python denoise.py --manifest manifests/b30.csv --out-dir outputs/b30
```

To split the work across jobs, add `--start 200 --limit 200` (slices 200 to 399).
For D45, pass the D45 folder name to `--kernel` and add `--dataset mayo-d45` in step 1.

## Notes

- Each slice is min-max normalized to [0, 1] on its own before training and scoring.
- Default settings are 3 stages, 2000 epochs, Adam (lr 1e-3, halved every 1000 epochs), and EMA plus an 8-fold flip and rotation ensemble at the end. Run `python denoise.py --help` for the rest.
- `--fixed-blend 0.8` uses a constant blend instead of the learned one.
- No GPU is used if none is found, but 2000 epochs on a 512x512 slice is slow on CPU.
- Other inputs work too. `.dcm`/`.ima` files (see `make_mayo_manifest.py`), RGB photos with `--channels rgb`, and FMD (see `make_fmd_manifest.py`).
