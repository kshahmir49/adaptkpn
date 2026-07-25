#!/usr/bin/env python3
"""
Create a manifest CSV for the final Range-Gated AdaptKPN vs F2N DICOM benchmark.

Assumed Mayo image-data folder layout:
  1mm B30/full_1mm/<PATIENT>/full_1mm/*.IMA
  1mm B30/quarter_1mm/<PATIENT>/quarter_1mm/*.IMA
  1mm D45/full_1mm_sharp/<PATIENT>/full_1mm_sharp/*.IMA
  1mm D45/quarter_1mm_sharp/<PATIENT>/quarter_1mm_sharp/*.IMA

Pairs are matched by z-position from ImagePositionPatient[2] when available,
then by InstanceNumber as a fallback. This script only creates the manifest;
it does not load pixel data.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom

# UIDs identified from your inventories. They are used as a safety filter.
UIDS = {
    "B30": {
        "L096": {
            "low": "1.3.12.2.1107.5.1.4.64291.30000015122322311130600000828",
            "full": "1.3.12.2.1107.5.1.4.64291.30000015122322311130600000002",
        },
        "L109": {
            "low": "1.3.12.2.1107.5.1.4.73013.30000015122323440599100002733",
            "full": "1.3.12.2.1107.5.1.4.73013.30000015122323440599100002412",
        },
        "L143": {
            "low": "1.3.12.2.1107.5.1.4.64291.30000015122302433284400001945",
            "full": "1.3.12.2.1107.5.1.4.64291.30000015122300005219400007837",
        },
        "L506": {
            "low": "1.3.12.2.1107.5.1.4.64291.30000015122302433284400002747",
            "full": "1.3.12.2.1107.5.1.4.64291.30000015122300005219400008639",
        },
    },
    "D45": {
        "L096": {
            "low": "1.3.12.2.1107.5.1.4.64291.30000016012200111078900001915",
            "full": "1.3.12.2.1107.5.1.4.64291.30000016012200111078900003074",
        },
        "L109": {
            "low": "1.3.12.2.1107.5.1.4.73013.30000016012119261090200000585",
            "full": "1.3.12.2.1107.5.1.4.73013.30000016012119261090200000133",
        },
        "L143": {
            "low": "1.3.12.2.1107.5.1.4.64291.30000016012200111078900004962",
            "full": "1.3.12.2.1107.5.1.4.64291.30000016012200111078900004137",
        },
        "L506": {
            "low": "1.3.12.2.1107.5.1.4.64291.30000016012200111078900012907",
            "full": "1.3.12.2.1107.5.1.4.64291.30000016012200111078900012164",
        },
    },
}


def candidate_files(folder: Path, recursive: bool = False) -> List[Path]:
    patterns = ["*.IMA", "*.ima", "*.dcm", "*.DCM"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(folder.rglob(pat) if recursive else folder.glob(pat))
    return sorted(set(files))


def z_key(ds) -> Optional[str]:
    try:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return f"z_{float(ipp[2]):.3f}"
    except Exception:
        return None
    return None


def inst_key(ds) -> Optional[str]:
    try:
        return f"inst_{int(getattr(ds, 'InstanceNumber')):05d}"
    except Exception:
        return None


def read_meta(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def collect_series(folder: Path, series_uid: Optional[str], recursive: bool) -> List[Dict[str, str]]:
    files = candidate_files(folder, recursive=recursive)
    if not files and not recursive:
        # Layout might be one level deeper than expected.
        files = candidate_files(folder, recursive=True)
    rows = []
    for i, path in enumerate(files):
        try:
            ds = read_meta(path)
            uid = str(getattr(ds, "SeriesInstanceUID", ""))
            if series_uid and uid != series_uid:
                continue
            zk = z_key(ds)
            ik = inst_key(ds)
            key = zk or ik
            if key is None:
                continue
            rows.append({
                "path": str(path),
                "series_uid": uid,
                "z_key": zk or "",
                "inst_key": ik or "",
                "pair_key": key,
                "z": str(float(ds.ImagePositionPatient[2])) if getattr(ds, "ImagePositionPatient", None) is not None else "",
                "instance": str(int(getattr(ds, "InstanceNumber", -1))) if getattr(ds, "InstanceNumber", None) is not None else "",
                "rows": str(int(getattr(ds, "Rows", -1))) if getattr(ds, "Rows", None) is not None else "",
                "cols": str(int(getattr(ds, "Columns", -1))) if getattr(ds, "Columns", None) is not None else "",
                "slope": str(getattr(ds, "RescaleSlope", 1)),
                "intercept": str(getattr(ds, "RescaleIntercept", 0)),
            })
        except Exception as e:
            print(f"Warning: failed metadata read: {path}: {e}")
    return rows


def patient_dirs(kernel: str, patient: str, b30_root: Path, d45_root: Path) -> Tuple[Path, Path]:
    if kernel == "B30":
        return (
            b30_root / "quarter_1mm" / patient / "quarter_1mm",
            b30_root / "full_1mm" / patient / "full_1mm",
        )
    if kernel == "D45":
        return (
            d45_root / "quarter_1mm_sharp" / patient / "quarter_1mm_sharp",
            d45_root / "full_1mm_sharp" / patient / "full_1mm_sharp",
        )
    raise ValueError(kernel)


def make_manifest(args):
    rows = []
    task_id = 0
    for kernel in args.kernels:
        for patient in args.patients:
            low_dir, full_dir = patient_dirs(kernel, patient, Path(args.b30_root), Path(args.d45_root))
            uid_info = UIDS.get(kernel, {}).get(patient, {}) if not args.no_uid_filter else {}
            low_uid = uid_info.get("low")
            full_uid = uid_info.get("full")
            print("\n" + "=" * 80)
            print(f"{kernel} {patient}")
            print(f"low dir : {low_dir}")
            print(f"full dir: {full_dir}")
            print(f"low uid : {low_uid or 'not filtered'}")
            print(f"full uid: {full_uid or 'not filtered'}")
            low_rows = collect_series(low_dir, low_uid, recursive=args.recursive)
            full_rows = collect_series(full_dir, full_uid, recursive=args.recursive)
            print(f"found low={len(low_rows)} full={len(full_rows)}")
            if not low_rows or not full_rows:
                print("WARNING: no files found for this patient/kernel; skipping")
                continue

            # Prefer z-keys if both sides have them; fallback to instance keys.
            low_map = {r["z_key"]: r for r in low_rows if r["z_key"]}
            full_map = {r["z_key"]: r for r in full_rows if r["z_key"]}
            common_keys = sorted(set(low_map) & set(full_map), key=lambda k: float(k.split("_")[1]))
            pair_mode = "z"
            if not common_keys:
                low_map = {r["inst_key"]: r for r in low_rows if r["inst_key"]}
                full_map = {r["inst_key"]: r for r in full_rows if r["inst_key"]}
                common_keys = sorted(set(low_map) & set(full_map))
                pair_mode = "instance"
            print(f"matched pairs={len(common_keys)} by {pair_mode}")

            for pair_idx, key in enumerate(common_keys):
                if args.max_per_patient is not None and pair_idx >= args.max_per_patient:
                    break
                lo = low_map[key]
                fu = full_map[key]
                rows.append({
                    "task_id": task_id,
                    "kernel": kernel,
                    "patient": patient,
                    "pair_idx": pair_idx,
                    "pair_key": key,
                    "pair_mode": pair_mode,
                    "low_path": lo["path"],
                    "full_path": fu["path"],
                    "low_series_uid": lo["series_uid"],
                    "full_series_uid": fu["series_uid"],
                    "low_z": lo["z"],
                    "full_z": fu["z"],
                    "low_instance": lo["instance"],
                    "full_instance": fu["instance"],
                    "window_min": args.window_min,
                    "window_max": args.window_max,
                })
                task_id += 1
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b30-root", default="1mm B30")
    p.add_argument("--d45-root", default="1mm D45")
    p.add_argument("--patients", nargs="+", default=["L096", "L109", "L506"])
    p.add_argument("--kernels", nargs="+", default=["B30", "D45"], choices=["B30", "D45"])
    p.add_argument("--window-min", type=float, default=-3000.0)
    p.add_argument("--window-max", type=float, default=3000.0)
    p.add_argument("--out", default="dicom_rangegated_manifest.csv")
    p.add_argument("--recursive", action="store_true", help="Recursively scan dose folders. Slower but useful if paths are nested differently.")
    p.add_argument("--no-uid-filter", action="store_true", help="Do not filter by known SeriesInstanceUID. Use only if folder contains one series.")
    p.add_argument("--max-per-patient", type=int, default=None, help="Debug option: only first N pairs per patient/kernel.")
    args = p.parse_args()

    rows = make_manifest(args)
    if not rows:
        raise RuntimeError("No manifest rows created. Check roots/patient folders.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\n" + "=" * 80)
    print(f"Wrote {len(rows)} rows to {out}")
    by = {}
    for r in rows:
        by[(r["kernel"], r["patient"])] = by.get((r["kernel"], r["patient"]), 0) + 1
    for (kernel, patient), n in sorted(by.items()):
        print(f"  {kernel} {patient}: {n}")


if __name__ == "__main__":
    main()
