#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import pydicom


def dicom_files(folder):
    folder = Path(folder)
    files = []
    for pattern in ("*.IMA", "*.ima", "*.dcm", "*.DCM"):
        files.extend(folder.rglob(pattern))
    return sorted(set(files))


def slice_key(path):
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)

    pos = getattr(ds, "ImagePositionPatient", None)
    if pos is not None and len(pos) >= 3:
        try:
            return f"z_{float(pos[2]):.3f}"
        except Exception:
            pass

    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        return f"inst_{int(inst):05d}"

    return None


def collect(folder):
    out = {}
    for path in dicom_files(folder):
        try:
            key = slice_key(path)
        except Exception:
            continue
        if key and key not in out:
            out[key] = str(path)
    return out


def main():
    p = argparse.ArgumentParser(description="Create a Mayo low/full-dose DICOM manifest.")
    p.add_argument("--low-dir", required=True)
    p.add_argument("--full-dir", required=True)
    p.add_argument("--dataset", required=True, choices=["mayo-b30", "mayo-d45"])
    p.add_argument("--patient", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-pairs", type=int, default=None)
    args = p.parse_args()

    low = collect(args.low_dir)
    full = collect(args.full_dir)
    keys = sorted(set(low) & set(full))

    if args.max_pairs is not None:
        keys = keys[:args.max_pairs]
    if not keys:
        raise RuntimeError("No matched DICOM pairs found.")

    rows = []
    for i, key in enumerate(keys):
        rows.append({
            "dataset": args.dataset,
            "patient": args.patient,
            "pair_idx": i,
            "pair_key": key,
            "low_path": low[key],
            "full_path": full[key],
            "window_min": -3000,
            "window_max": 3000,
            "channels": "gray",
            "tag": f"{args.dataset}_{args.patient}_{i:04d}",
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
