#!/usr/bin/env python3
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from skimage.metrics import structural_similarity
import torch

from adaptkpn import frozen_args, train_adaptkpn


def load_ct(path: str, normalization: str):
    ds = pydicom.dcmread(path, force=True)
    raw = ds.pixel_array.astype(np.float32)
    if normalization == "fixed4095":
        arr = np.clip(raw / 4095.0, 0.0, 1.0).astype(np.float32)
    elif normalization == "minmax":
        lo = float(np.min(raw)); hi = float(np.max(raw))
        if hi <= lo:
            raise ValueError(f"Degenerate slice range {lo}..{hi} for {path}")
        arr = ((raw - lo) / (hi - lo)).astype(np.float32)
    else:
        raise ValueError(normalization)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


def make_even(x):
    h, w = x.shape[-2:]
    return x[..., :h - (h % 2), :w - (w % 2)].contiguous()


def psnr(x, y):
    mse = torch.mean((x - y) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def ssim(x, y):
    a = x.detach().clamp(0, 1).cpu().numpy()[0, 0]
    b = y.detach().clamp(0, 1).cpu().numpy()[0, 0]
    return float(structural_similarity(b, a, data_range=1.0))


def save_gray(x, path):
    arr = (x.detach().clamp(0, 1).cpu().numpy()[0, 0] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_rows(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Run final AdaptKPN on paired CT DICOM/IMA slices")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--normalization", choices=["fixed4095", "minmax"], required=True)
    ap.add_argument("--batch-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out-dir", default="outputs/ct")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.manifest).sort_values("task_id").reset_index(drop=True)
    start = args.batch_index * args.batch_size
    end = min(start + args.batch_size, len(df))
    batch = df.iloc[start:end]
    if batch.empty:
        raise SystemExit("Selected CT batch is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for _, src in batch.iterrows():
        row = src.to_dict()
        row.update(status="ok", error="")
        try:
            noisy = make_even(load_ct(src.low_path, args.normalization)).to(device)
            clean = make_even(load_ct(src.full_path, args.normalization)).to(device)
            if noisy.shape != clean.shape:
                raise ValueError(f"Shape mismatch {tuple(noisy.shape)} vs {tuple(clean.shape)}")
            row["noisy_psnr"] = psnr(noisy, clean)
            row["noisy_ssim"] = ssim(noisy, clean)

            cfg = frozen_args(seed=args.seed + int(src.task_id), epochs=args.epochs, amp=not args.no_amp)
            sync(device)
            t0 = time.perf_counter()
            model, final_loss = train_adaptkpn(noisy, cfg, device)
            with torch.no_grad():
                den = model(noisy).clamp(0, 1).contiguous()
            sync(device)
            row["adaptkpn_seconds"] = time.perf_counter() - t0
            row["adaptkpn_psnr"] = psnr(den, clean)
            row["adaptkpn_ssim"] = ssim(den, clean)
            row["adaptkpn_final_loss"] = final_loss
            row["adaptkpn_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            row["adaptkpn_sigma_r_values"] = ";".join(f"{v:.8f}" for v in model.sigma_r_values())
            row["normalization"] = args.normalization
            row["epochs"] = args.epochs

            if args.save_images:
                stem = f"task_{int(src.task_id):05d}"
                save_gray(noisy, out_dir / f"{stem}_noisy.png")
                save_gray(den, out_dir / f"{stem}_adaptkpn.png")
                save_gray(clean, out_dir / f"{stem}_gt.png")
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        rows.append(row)

    out_csv = out_dir / f"ct_batch_{args.batch_index}.csv"
    write_rows(out_csv, rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
