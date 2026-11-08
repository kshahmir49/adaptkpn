#!/usr/bin/env python3
import argparse
import glob
import pandas as pd

LABEL = {
    "Confocal_BPAE": "Confocal BPAE",
    "TwoPhoton_BPAE": "Photon BPAE",
    "TwoPhoton_MICE": "Photon Mice",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--out-prefix", default="fmd_final")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit("No FMD result CSV files found")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ok = df[df.status.eq("ok")].copy()

    print("Status")
    print(df.status.value_counts(dropna=False))

    metrics = [
        "noisy_psnr",
        "noisy_ssim",
        "adaptkpn_psnr",
        "adaptkpn_ssim",
        "adaptkpn_seconds",
    ]

    by = ok.groupby("category")[metrics].mean(numeric_only=True)
    by.insert(0, "Image", [LABEL.get(x, x) for x in by.index])

    print("\nCategory means")
    print(by.round(4).to_string(index=False))

    category_mean = ok.groupby("category")[metrics].mean(numeric_only=True).mean(axis=0)
    print("\nSimple three-category average")
    print(category_mean.round(4).to_string())

    ok.to_csv(f"{args.out_prefix}_per_image.csv", index=False)
    by.reset_index(drop=True).to_csv(f"{args.out_prefix}_by_category.csv", index=False)
    pd.DataFrame([category_mean]).to_csv(f"{args.out_prefix}_average.csv", index=False)

    print(f"\nSaved {args.out_prefix}_per_image.csv")
    print(f"Saved {args.out_prefix}_by_category.csv")
    print(f"Saved {args.out_prefix}_average.csv")


if __name__ == "__main__":
    main()
