# Mayo CT reproduction

## 1. Create manifest

```bash
python scripts/mayo/make_dicom_manifest_rangegated.py \
  --b30-root "1mm B30" \
  --d45-root "1mm D45" \
  --patients L096 L109 L506 \
  --kernels B30 D45 \
  --out manifests/dicom_rangegated_manifest.csv
```

The manifest script matches low-dose and full-dose DICOM slices using z-position when available, with instance number as fallback.

## 2. Run benchmark

```bash
python scripts/mayo/run_rangegated_dicom_manifest_batch.py \
  --manifest manifests/dicom_rangegated_manifest.csv \
  --batch-index 0 \
  --batch-size 50 \
  --out-csv results/mayo_batch_000.csv
```

For Slurm arrays, map the array task ID to `--batch-index`.

## 3. Summarize

```bash
python scripts/mayo/summarize_rangegated_dicom_results.py \
  "results/mayo_batch_*.csv" \
  --out-dir results/mayo_summary
```

Do not commit real manifests with absolute HPC paths.
