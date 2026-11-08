#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

TARGET = ["TwoPhoton_BPAE", "TwoPhoton_MICE", "Confocal_BPAE"]
EXTS = {".png", ".tif", ".tiff"}


def parse_name(stem: str):
    parts = stem.split("_")
    category = "_".join(parts[:2]) if len(parts) >= 2 else stem
    channel = parts[2] if category != "TwoPhoton_MICE" and len(parts) >= 4 else ""
    sample_id = parts[-1] if parts and parts[-1].isdigit() else ""
    return category, channel, sample_id


def main():
    ap = argparse.ArgumentParser(description="Create an FMD raw/GT manifest")
    ap.add_argument("--root", required=True, help="Dataset root containing raw/ and gt/")
    ap.add_argument("--categories", nargs="+", default=TARGET)
    ap.add_argument("--out", default="fmd_manifest.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    raw_dir = root / "raw"
    gt_dir = root / "gt"
    if not raw_dir.is_dir() or not gt_dir.is_dir():
        raise SystemExit(f"Expected {raw_dir} and {gt_dir}")

    wanted = set(args.categories)
    rows = []
    for raw_path in sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS):
        category, channel, sample_id = parse_name(raw_path.stem)
        if category not in wanted:
            continue
        rel = raw_path.relative_to(raw_dir)
        gt_path = gt_dir / rel
        if not gt_path.exists():
            candidates = [gt_dir / rel.parent / f"{raw_path.stem}{ext}" for ext in EXTS]
            gt_path = next((p for p in candidates if p.exists()), None)
        if gt_path is None or not Path(gt_path).exists():
            raise SystemExit(f"Missing GT for {raw_path}")
        rows.append({
            "task_id": len(rows),
            "category": category,
            "channel": channel,
            "sample_id": sample_id,
            "raw_path": str(raw_path.resolve()),
            "gt_path": str(Path(gt_path).resolve()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No requested FMD pairs found")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("category").size().to_string())
    print(f"Total {len(df)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
