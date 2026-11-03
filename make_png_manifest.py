"""Build a manifest of matched low-dose / full-dose PNG slices.

Expected layout (the Preprocessed_512x512 release):
  ROOT/<low dose folder>/<kernel folder>/<patient>/*.png
  ROOT/<full dose folder>/<kernel folder>/<patient>/*.png
"""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Create a low/full-dose PNG manifest.")
    p.add_argument("--root", required=True, help="e.g. Preprocessed_512x512/512")
    p.add_argument("--low", default="Quarter Dose/1mm")
    p.add_argument("--full", default="Full Dose/1mm")
    p.add_argument("--kernel", required=True, help='e.g. "Soft Kernel (B30)" or the D45 kernel folder name')
    p.add_argument("--patients", nargs="+", required=True, help="e.g. L096 L506 L109")
    p.add_argument("--dataset", default="mayo-b30", choices=["mayo-b30", "mayo-d45"])
    p.add_argument("--limit-per-patient", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    for pid in args.patients:
        low = sorted((root / args.low / args.kernel / pid).glob("*.png"))
        full = sorted((root / args.full / args.kernel / pid).glob("*.png"))
        n = min(len(low), len(full))
        if args.limit_per_patient:
            n = min(n, args.limit_per_patient)
        if len(low) != len(full):
            print(f"Warning {pid}: low={len(low)} full={len(full)}, pairing first {n}")
        print(f"{pid}: {n} pairs")
        for i in range(n):
            rows.append({
                "dataset": args.dataset,
                "tag": f"{pid}_{i + 1:04d}",
                "noisy": str(low[i]),
                "target": str(full[i]),
                "channels": "gray",
            })

    if not rows:
        raise SystemExit("No pairs found. Check --root, --kernel and --patients.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
