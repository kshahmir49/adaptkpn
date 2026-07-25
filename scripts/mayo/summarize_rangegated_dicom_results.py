#!/usr/bin/env python3
"""Summarize Range-Gated DICOM benchmark batch CSVs."""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def mean_std(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def summarize_group(df, keys):
    rows = []
    for name, g in df.groupby(keys, dropna=False):
        if not isinstance(name, tuple):
            name = (name,)
        row = {k: v for k, v in zip(keys, name)}
        row["n"] = len(g)
        for prefix in ["noisy", "f2n", "ours_raw", "ours_blend"]:
            for metric in ["psnr", "ssim"]:
                col = f"{prefix}_{metric}"
                if col in g.columns:
                    m, s = mean_std(g[col])
                    row[f"{col}_mean"] = m
                    row[f"{col}_std"] = s
        if "f2n_psnr" in g.columns and "ours_blend_psnr" in g.columns:
            dpsnr = pd.to_numeric(g["ours_blend_psnr"], errors="coerce") - pd.to_numeric(g["f2n_psnr"], errors="coerce")
            dssim = pd.to_numeric(g["ours_blend_ssim"], errors="coerce") - pd.to_numeric(g["f2n_ssim"], errors="coerce")
            row["ours_minus_f2n_psnr_mean"] = float(dpsnr.mean())
            row["ours_minus_f2n_ssim_mean"] = float(dssim.mean())
            row["ours_psnr_wins_vs_f2n"] = int((dpsnr > 0).sum())
            row["ours_ssim_wins_vs_f2n"] = int((dssim > 0).sum())
        if "noisy_psnr" in g.columns and "ours_blend_psnr" in g.columns:
            dpsnr_n = pd.to_numeric(g["ours_blend_psnr"], errors="coerce") - pd.to_numeric(g["noisy_psnr"], errors="coerce")
            dssim_n = pd.to_numeric(g["ours_blend_ssim"], errors="coerce") - pd.to_numeric(g["noisy_ssim"], errors="coerce")
            row["ours_minus_noisy_psnr_mean"] = float(dpsnr_n.mean())
            row["ours_minus_noisy_ssim_mean"] = float(dssim_n.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    p.add_argument("--out-dir", default="rangegated_dicom_summary")
    p.add_argument("--combined-out", default=None)
    args = p.parse_args()

    files = []
    for pat in args.inputs:
        matches = sorted(glob.glob(pat))
        if matches:
            files.extend(matches)
        else:
            files.append(pat)
    files = sorted(set(files))
    if not files:
        raise RuntimeError("No input CSVs found")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                df["source_csv"] = f
                dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")
    if not dfs:
        raise RuntimeError("No non-empty CSV rows found")
    df = pd.concat(dfs, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_out = Path(args.combined_out) if args.combined_out else out_dir / "combined_results.csv"
    df.to_csv(combined_out, index=False)

    print("Files:", len(files))
    print("Rows:", len(df))
    if "status" in df.columns:
        print("Status counts:")
        print(df["status"].value_counts(dropna=False))
        df_ok = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    else:
        df_ok = df.copy()
    print("OK rows:", len(df_ok))

    if len(df_ok) == 0:
        print("No OK rows. Writing combined file only.")
        return

    # Save summaries.
    overall = summarize_group(df_ok.assign(overall="ALL"), ["overall"])
    by_kernel = summarize_group(df_ok, ["kernel"])
    by_kernel_patient = summarize_group(df_ok, ["kernel", "patient"])

    overall.to_csv(out_dir / "summary_overall.csv", index=False)
    by_kernel.to_csv(out_dir / "summary_by_kernel.csv", index=False)
    by_kernel_patient.to_csv(out_dir / "summary_by_kernel_patient.csv", index=False)

    print("\nSummary by kernel:")
    cols = [
        "kernel", "n",
        "noisy_psnr_mean", "f2n_psnr_mean", "ours_blend_psnr_mean", "ours_minus_f2n_psnr_mean",
        "noisy_ssim_mean", "f2n_ssim_mean", "ours_blend_ssim_mean", "ours_minus_f2n_ssim_mean",
        "ours_psnr_wins_vs_f2n", "ours_ssim_wins_vs_f2n",
    ]
    show_cols = [c for c in cols if c in by_kernel.columns]
    print(by_kernel[show_cols].to_string(index=False))
    print(f"\nWrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
