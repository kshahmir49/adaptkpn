#!/usr/bin/env python3
import argparse
import glob
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--out", default="ct_summary.csv")
    args = ap.parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit("No CT result CSV files found")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ok = df[df.status.eq("ok")].copy()
    metrics = ["noisy_psnr", "noisy_ssim", "adaptkpn_psnr", "adaptkpn_ssim", "adaptkpn_seconds"]
    print("Status")
    print(df.status.value_counts(dropna=False))
    print("
Overall")
    print(ok[metrics].mean(numeric_only=True).round(6))
    group_cols = [c for c in ["kernel", "patient"] if c in ok.columns]
    if group_cols:
        by = ok.groupby(group_cols)[metrics].mean(numeric_only=True).reset_index()
        print("
By group")
        print(by.to_string(index=False))
        by.to_csv(args.out, index=False)
    else:
        pd.DataFrame([ok[metrics].mean(numeric_only=True)]).to_csv(args.out, index=False)
    print(f"
Saved {args.out}")


if __name__ == "__main__":
    main()
