import argparse
import csv
from pathlib import Path


def find_pairs(root, raw_subdir, gt_subdir, exts):
    root = Path(root)
    raw_dir = root / raw_subdir
    gt_dir = root / gt_subdir

    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)

    raw_files = []
    for ext in exts:
        raw_files.extend(raw_dir.rglob(f"*{ext}"))

    pairs = []
    for raw in sorted(raw_files):
        rel = raw.relative_to(raw_dir)
        gt = gt_dir / rel
        if not gt.exists():
            matches = [gt_dir / rel.parent / f"{raw.stem}{ext}" for ext in exts]
            gt = next((p for p in matches if p.exists()), None)
        if gt is not None and gt.exists():
            pairs.append((raw, gt))
    return pairs


def main():
    p = argparse.ArgumentParser(description="Create an FMD raw/GT manifest.")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-subdir", default="raw")
    p.add_argument("--gt-subdir", default="gt")
    p.add_argument("--channels", choices=["gray", "rgb"], default="gray")
    p.add_argument("--exts", nargs="+", default=[".png", ".tif", ".tiff"])
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    pairs = find_pairs(args.root, args.raw_subdir, args.gt_subdir, tuple(args.exts))
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    if not pairs:
        raise RuntimeError("No FMD raw/GT pairs found.")

    rows = []
    for i, (raw, gt) in enumerate(pairs):
        rows.append({
            "dataset": "fmd",
            "image": raw.name,
            "raw_path": str(raw),
            "gt_path": str(gt),
            "channels": args.channels,
            "tag": raw.stem,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
