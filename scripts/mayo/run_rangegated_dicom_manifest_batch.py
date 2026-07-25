#!/usr/bin/env python3
"""
Run the final Range-Gated AdaptKPN and F2N baseline from a DICOM manifest.

This script processes one slice per manifest row:
  DICOM -> HU -> clip [-3000,3000] -> normalize [0,1]
  train Range-Gated AdaptKPN on the low-dose slice
  train F2N baseline on the same low-dose slice
  evaluate both against the full-dose slice

Designed for Slurm arrays via --batch-index and --batch-size.
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import torch


def import_model_module(v4_path: str):
    v4_path = os.path.abspath(v4_path)
    v4_dir = os.path.dirname(v4_path)
    if v4_dir and v4_dir not in sys.path:
        sys.path.insert(0, v4_dir)
    spec = importlib.util.spec_from_file_location("rangegated_adaptkpn", v4_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import script from {v4_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_args(cli) -> SimpleNamespace:
    return SimpleNamespace(
        seed=cli.seed,
        amp=cli.amp,
        compile=cli.compile,
        epochs=cli.epochs,
        lr=cli.lr,
        optimizer="adamw",
        weight_decay=cli.weight_decay,
        scheduler="onecycle",
        step_size=500,
        gamma=0.5,
        loss_mode="f2n",
        adapt_lambda_edge=cli.adapt_lambda_edge,
        kpn_chan=cli.kpn_chan,
        kpn_k=cli.kpn_k,
        kpn_stages=cli.kpn_stages,
        smooth_mix=cli.smooth_mix,
        use_range_gate=True,
        range_sigma_init=cli.range_sigma_init,
        range_sigma_min=cli.range_sigma_min,
        range_sigma_max=cli.range_sigma_max,
        edge_weight=0.02,
        use_ema=cli.use_ema,
        ema_decay=cli.ema_decay,
        self_ensemble=cli.self_ensemble,
        poisson_weight=0.0,
        poisson_peak=1000.0,
        poisson_thin_p=0.5,
        f2n_epochs=cli.f2n_epochs,
        f2n_stages=cli.f2n_stages,
        f2n_patch_size=cli.f2n_patch_size,
        f2n_lr=cli.f2n_lr,
        f2n_weight_decay=cli.f2n_weight_decay,
        f2n_lambda_edge=cli.f2n_lambda_edge,
        print_every=cli.print_every,
    )


def load_dicom_windowed(path: str, window_min: float, window_max: float) -> torch.Tensor:
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    x = (np.clip(hu, window_min, window_max) - window_min) / (window_max - window_min)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0).contiguous()


def write_rows(path: str, rows: list):
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stable superset of fields in order of first appearance.
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_rows(df: pd.DataFrame, batch_index: int, batch_size: int, max_rows=None):
    df = df.sort_values("task_id").reset_index(drop=True)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    start = batch_index * batch_size
    end = min(start + batch_size, len(df))
    if start >= len(df):
        return df.iloc[0:0].copy(), start, end, len(df)
    return df.iloc[start:end].copy(), start, end, len(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-rows", type=int, default=None, help="Debug: only process first N manifest rows before batching.")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--v4-path", default="scripts/denoise_one_image_rangegated_hpc.py")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-f2n", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0)

    # Final frozen settings.
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--adapt-lambda-edge", type=float, default=350.0)
    p.add_argument("--kpn-chan", type=int, default=16)
    p.add_argument("--kpn-k", type=int, default=5)
    p.add_argument("--kpn-stages", type=int, default=3)
    p.add_argument("--smooth-mix", type=float, default=0.75)
    p.add_argument("--range-sigma-init", type=float, default=0.08)
    p.add_argument("--range-sigma-min", type=float, default=0.005)
    p.add_argument("--range-sigma-max", type=float, default=0.25)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--self-ensemble", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=123)

    # F2N settings.
    p.add_argument("--f2n-epochs", type=int, default=500)
    p.add_argument("--f2n-stages", type=int, default=2)
    p.add_argument("--f2n-patch-size", type=int, default=8)
    p.add_argument("--f2n-lr", type=float, default=1e-3)
    p.add_argument("--f2n-weight-decay", type=float, default=0.01)
    p.add_argument("--f2n-lambda-edge", type=float, default=350.0)
    p.add_argument("--print-every", type=int, default=100)
    args_cli = p.parse_args()

    mod = import_model_module(args_cli.v4_path)
    model_args = make_args(args_cli)
    device = torch.device(args_cli.device if args_cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    print("=" * 80)
    print("Range-Gated DICOM batch")
    print(f"manifest: {args_cli.manifest}")
    print(f"batch index/size: {args_cli.batch_index}/{args_cli.batch_size}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET')}")
        print(f"cuda device count visible={torch.cuda.device_count()}")
        print(f"cuda device name={torch.cuda.get_device_name(0)}")
    print(f"range_sigma_init={args_cli.range_sigma_init}, alpha={args_cli.alpha}, skip_f2n={args_cli.skip_f2n}")

    manifest = pd.read_csv(args_cli.manifest)
    batch, start, end, total = select_rows(manifest, args_cli.batch_index, args_cli.batch_size, args_cli.max_rows)
    print(f"selected rows: {len(batch)}  global range [{start}, {end}) of {total}")
    if len(batch) == 0:
        print("No rows for this batch. Writing empty CSV with no rows.")
        Path(args_cli.out_csv).parent.mkdir(parents=True, exist_ok=True)
        batch.head(0).to_csv(args_cli.out_csv, index=False)
        return

    rows = []
    for local_i, row0 in enumerate(batch.to_dict(orient="records")):
        print("\n" + "-" * 80)
        print(f"[{local_i+1}/{len(batch)}] task_id={row0.get('task_id')} {row0.get('kernel')} {row0.get('patient')} pair_idx={row0.get('pair_idx')} key={row0.get('pair_key')}")
        result = dict(row0)
        result.update({
            "status": "ok",
            "error": "",
            "alpha": args_cli.alpha,
            "range_sigma_init": args_cli.range_sigma_init,
            "ours_epochs": args_cli.epochs,
            "f2n_epochs": args_cli.f2n_epochs if not args_cli.skip_f2n else 0,
        })
        try:
            wmin = float(row0.get("window_min", -3000.0))
            wmax = float(row0.get("window_max", 3000.0))
            low = load_dicom_windowed(str(row0["low_path"]), wmin, wmax).to(device)
            full = load_dicom_windowed(str(row0["full_path"]), wmin, wmax).to(device)
            if tuple(low.shape) != tuple(full.shape):
                raise ValueError(f"shape mismatch: {tuple(low.shape)} vs {tuple(full.shape)}")
            low_train, pad_hw = mod.pad_to_even(low)
            result["noisy_psnr"] = mod.compute_psnr(low, full)
            result["noisy_ssim"] = mod.compute_ssim(low, full)
            print(f"Noisy      PSNR={result['noisy_psnr']:.2f}  SSIM={result['noisy_ssim']:.4f}")

            t0 = time.time()
            model, final_loss = mod.train_adaptkpn_d45(low_train, model_args, device)
            with torch.no_grad():
                den_raw = mod.denoise_tta(model, low_train, use_tta=args_cli.self_ensemble).clamp(0, 1).contiguous()
            den_raw = mod.unpad_even(den_raw, pad_hw)
            result["ours_seconds"] = time.time() - t0
            result["ours_final_loss"] = final_loss
            result["ours_raw_psnr"] = mod.compute_psnr(den_raw, full)
            result["ours_raw_ssim"] = mod.compute_ssim(den_raw, full)
            if hasattr(model, "sigma_r_values"):
                result["ours_sigma_r_values"] = ";".join(f"{v:.6f}" for v in model.sigma_r_values())
            else:
                result["ours_sigma_r_values"] = ""
            blend = ((1.0 - args_cli.alpha) * low + args_cli.alpha * den_raw).clamp(0, 1)
            result["ours_blend_psnr"] = mod.compute_psnr(blend, full)
            result["ours_blend_ssim"] = mod.compute_ssim(blend, full)
            print(f"Ours raw   PSNR={result['ours_raw_psnr']:.2f}  SSIM={result['ours_raw_ssim']:.4f} ({result['ours_seconds']:.1f}s)")
            print(f"Ours blend PSNR={result['ours_blend_psnr']:.2f}  SSIM={result['ours_blend_ssim']:.4f} alpha={args_cli.alpha:g}")

            if not args_cli.skip_f2n:
                print("Running F2N...")
                t0 = time.time()
                den_f2n = mod.run_f2n(low_train, model_args, device)
                den_f2n = mod.unpad_even(den_f2n, pad_hw)
                result["f2n_seconds"] = time.time() - t0
                result["f2n_psnr"] = mod.compute_psnr(den_f2n, full)
                result["f2n_ssim"] = mod.compute_ssim(den_f2n, full)
                print(f"F2N        PSNR={result['f2n_psnr']:.2f}  SSIM={result['f2n_ssim']:.4f} ({result['f2n_seconds']:.1f}s)")
        except Exception as e:
            result["status"] = "error"
            result["error"] = repr(e)
            print(f"ERROR: {e}")
        rows.append(result)
        write_rows(args_cli.out_csv, rows)
        print(f"Wrote {args_cli.out_csv}")

    ok = [r for r in rows if r.get("status") == "ok"]
    print("\n" + "=" * 80)
    print(f"Batch finished: ok={len(ok)} / {len(rows)}")
    if ok:
        for col in ["noisy_psnr", "f2n_psnr", "ours_blend_psnr", "noisy_ssim", "f2n_ssim", "ours_blend_ssim"]:
            vals = [float(r[col]) for r in ok if col in r and r.get(col, "") != ""]
            if vals:
                print(f"{col}: {float(np.mean(vals)):.4f}")


if __name__ == "__main__":
    main()
