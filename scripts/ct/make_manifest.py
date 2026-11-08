#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import pydicom

EXTS = {".ima", ".dcm"}


def collect(folder: Path, series_uid: str | None):
    rows = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", ""))
            if series_uid and uid != series_uid:
                continue
            ipp = getattr(ds, "ImagePositionPatient", None)
            z = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else None
            inst = int(getattr(ds, "InstanceNumber")) if getattr(ds, "InstanceNumber", None) is not None else None
            rows.append({"path": str(path.resolve()), "series_uid": uid, "z": z, "instance": inst})
        except Exception as exc:
            print(f"Warning failed metadata read {path} {exc}")
    return rows


def pair_rows(low_rows, full_rows):
    low_z = {round(r["z"], 3): r for r in low_rows if r["z"] is not None}
    full_z = {round(r["z"], 3): r for r in full_rows if r["z"] is not None}
    common = sorted(set(low_z) & set(full_z))
    if common:
        return [("z", k, low_z[k], full_z[k]) for k in common]
    low_i = {r["instance"]: r for r in low_rows if r["instance"] is not None}
    full_i = {r["instance"]: r for r in full_rows if r["instance"] is not None}
    common = sorted(set(low_i) & set(full_i))
    return [("instance", k, low_i[k], full_i[k]) for k in common]


def main():
    ap = argparse.ArgumentParser(description="Create a paired low-dose/full-dose CT manifest")
    ap.add_argument("--low-dir", required=True)
    ap.add_argument("--full-dir", required=True)
    ap.add_argument("--patient", default="")
    ap.add_argument("--kernel", choices=["B30", "D45"], required=True)
    ap.add_argument("--low-series-uid", default=None)
    ap.add_argument("--full-series-uid", default=None)
    ap.add_argument("--expected", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    low = collect(Path(args.low_dir), args.low_series_uid)
    full = collect(Path(args.full_dir), args.full_series_uid)
    pairs = pair_rows(low, full)
    if not pairs:
        raise SystemExit("No matched CT pairs found")
    if args.expected is not None and len(pairs) != args.expected:
        raise SystemExit(f"Expected {args.expected} pairs but found {len(pairs)}")

    rows = []
    for task_id, (mode, key, lo, fu) in enumerate(pairs):
        rows.append({
            "task_id": task_id,
            "patient": args.patient,
            "kernel": args.kernel,
            "pair_mode": mode,
            "pair_key": key,
            "low_path": lo["path"],
            "full_path": fu["path"],
            "low_series_uid": lo["series_uid"],
            "full_series_uid": fu["series_uid"],
            "low_instance": lo["instance"],
            "full_instance": fu["instance"],
            "low_z": lo["z"],
            "full_z": fu["z"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
