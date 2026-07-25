#!/usr/bin/env python3
"""
Run Range-Gated AdaptKPN and F2N on DICOM-prepared .npy low/full image pairs.

Expected input folders are produced by raw_dicom_prepare_v2.py eval-pair:
  prepared_dicom/.../low_npy/*.npy
  prepared_dicom/.../full_npy/*.npy

This script imports the model/loss functions from denoise_one_image_d45_v4_hpc.py,
so keep that file and filter2noise.py in the same working directory.
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
import torch


def import_v4(v4_path: str):
    v4_path = os.path.abspath(v4_path)
    v4_dir = os.path.dirname(v4_path)
    if v4_dir and v4_dir not in sys.path:
        sys.path.insert(0, v4_dir)
    spec = importlib.util.spec_from_file_location("adaptkpn_v4_hpc", v4_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import v4 script from {v4_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_npy01(path: str) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D npy image, got {arr.shape} at {path}")
    # These arrays should already be [0,1] from raw_dicom_prepare_v2.py.
    # Clip only for numerical safety, not contrast normalization.
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


def pair_npy_files(low_dir: str, full_dir: str):
    low_files = sorted(Path(low_dir).glob("*.npy"))
    full_files = sorted(Path(full_dir).glob("*.npy"))
    low_map = {p.name: p for p in low_files}
    full_map = {p.name: p for p in full_files}
    names = sorted(set(low_map) & set(full_map))
    if not names:
        # Fallback: raw_dicom_prepare_v2 uses same index prefix. Match by first 4 digits.
        low_idx = {p.name.split("_")[0]: p for p in low_files}
        full_idx = {p.name.split("_")[0]: p for p in full_files}
        keys = sorted(set(low_idx) & set(full_idx), key=lambda x: int(x) if x.isdigit() else x)
        pairs = [(low_idx[k], full_idx[k], k) for k in keys]
    else:
        pairs = [(low_map[n], full_map[n], n) for n in names]
    if not pairs:
        raise RuntimeError(f"No matched .npy files found in {low_dir} and {full_dir}")
    return pairs


def make_v4_args(cli) -> SimpleNamespace:
    # Attributes required by train_adaptkpn_d45(), run_f2n(), and helpers in the V4 script.
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
        use_range_gate=cli.use_range_gate,
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


def write_rows(path: str, rows: list):
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, alpha_values):
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        print("No successful rows to summarize.")
        return
    def mean(col):
        vals = [float(r[col]) for r in ok if r.get(col, "") not in ("", None)]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nSummary over successful rows")
    print(f"  n = {len(ok)}")
    print(f"  Noisy: PSNR={mean('noisy_psnr'):.2f}  SSIM={mean('noisy_ssim'):.4f}")
    if "f2n_psnr" in ok[0]:
        print(f"  F2N:   PSNR={mean('f2n_psnr'):.2f}  SSIM={mean('f2n_ssim'):.4f}")
    print(f"  Ours raw: PSNR={mean('ours_raw_psnr'):.2f}  SSIM={mean('ours_raw_ssim'):.4f}")
    for a in alpha_values:
        tag = str(a).replace('.', 'p')
        print(f"  Ours blend alpha={a:g}: PSNR={mean('ours_blend_psnr_a'+tag):.2f}  SSIM={mean('ours_blend_ssim_a'+tag):.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Run AdaptKPN-V4 and F2N on prepared .npy low/full pairs.")
    p.add_argument("--v4-path", default="scripts/denoise_one_image_rangegated_hpc.py")
    p.add_argument("--low-npy-dir", required=True)
    p.add_argument("--full-npy-dir", required=True)
    p.add_argument("--indices", nargs="*", type=int, default=None, help="0-based pair indices to run. Example: --indices 0 10 25")
    p.add_argument("--max-slices", type=int, default=None, help="Use first N pairs after optional index filtering.")
    p.add_argument("--alphas", nargs="*", type=float, default=[0.75], help="Residual blend alpha values to evaluate.")
    p.add_argument("--out-csv", default="npy_ours_f2n_results.csv")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--out-dir", default="./npy_ours_f2n_outputs")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--compile", action="store_true")

    # Ours frozen defaults
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--adapt-lambda-edge", type=float, default=350.0)
    p.add_argument("--kpn-chan", type=int, default=16)
    p.add_argument("--kpn-k", type=int, default=5)
    p.add_argument("--kpn-stages", type=int, default=3)
    p.add_argument("--smooth-mix", type=float, default=0.75)
    p.add_argument("--range-gate", dest="use_range_gate", action="store_true", default=True, help="Enable intensity/range gate inside AdaptKPN.")
    p.add_argument("--no-range-gate", dest="use_range_gate", action="store_false")
    p.add_argument("--range-sigma-init", type=float, default=0.05, help="Initial learnable range sigma in [0,1] units. Try 0.03, 0.05, 0.08.")
    p.add_argument("--range-sigma-min", type=float, default=0.005)
    p.add_argument("--range-sigma-max", type=float, default=0.25)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--self-ensemble", action="store_true")

    # F2N defaults
    p.add_argument("--skip-f2n", action="store_true", help="Only run AdaptKPN; skip F2N.")
    p.add_argument("--f2n-epochs", type=int, default=500)
    p.add_argument("--f2n-stages", type=int, default=2)
    p.add_argument("--f2n-patch-size", type=int, default=8)
    p.add_argument("--f2n-lr", type=float, default=1e-3)
    p.add_argument("--f2n-weight-decay", type=float, default=0.01)
    p.add_argument("--f2n-lambda-edge", type=float, default=350.0)

    p.add_argument("--print-every", type=int, default=100)
    return p.parse_args()


def main():
    cli = parse_args()
    mod = import_v4(cli.v4_path)
    args = make_v4_args(cli)

    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    pairs = pair_npy_files(cli.low_npy_dir, cli.full_npy_dir)
    if cli.indices is not None:
        wanted = set(cli.indices)
        pairs = [p for i, p in enumerate(pairs) if i in wanted]
    if cli.max_slices is not None:
        pairs = pairs[:cli.max_slices]
    if not pairs:
        raise RuntimeError("No pairs selected.")

    Path(cli.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"selected pairs: {len(pairs)}")
    print(f"alphas: {cli.alphas}")
    print(f"range gate: {cli.use_range_gate}, sigma_init={cli.range_sigma_init}, sigma_bounds=[{cli.range_sigma_min}, {cli.range_sigma_max}]")

    rows = []
    for run_idx, (low_path, full_path, key) in enumerate(pairs):
        print("\n" + "=" * 80)
        print(f"[{run_idx+1}/{len(pairs)}] {key}")
        print(f"low : {low_path}")
        print(f"full: {full_path}")

        row = {
            "run_idx": run_idx,
            "key": key,
            "low_path": str(low_path),
            "full_path": str(full_path),
            "status": "ok",
            "error": "",
        }
        try:
            low = load_npy01(str(low_path)).to(device)
            full = load_npy01(str(full_path)).to(device)
            if low.shape != full.shape:
                raise ValueError(f"shape mismatch: {tuple(low.shape)} vs {tuple(full.shape)}")

            low_train, pad_hw = mod.pad_to_even(low)
            row["noisy_psnr"] = mod.compute_psnr(low, full)
            row["noisy_ssim"] = mod.compute_ssim(low, full)
            print(f"Noisy      PSNR={row['noisy_psnr']:.2f}  SSIM={row['noisy_ssim']:.4f}")

            t0 = time.time()
            model, final_loss = mod.train_adaptkpn_d45(low_train, args, device)
            with torch.no_grad():
                den_raw = mod.denoise_tta(model, low_train, use_tta=cli.self_ensemble).clamp(0, 1).contiguous()
            den_raw = mod.unpad_even(den_raw, pad_hw)
            row["ours_seconds"] = time.time() - t0
            row["ours_final_loss"] = final_loss
            if hasattr(model, "sigma_r_values"):
                row["ours_sigma_r_values"] = ";".join(f"{v:.6f}" for v in model.sigma_r_values())
            else:
                row["ours_sigma_r_values"] = ""
            row["ours_raw_psnr"] = mod.compute_psnr(den_raw, full)
            row["ours_raw_ssim"] = mod.compute_ssim(den_raw, full)
            print(f"Ours raw   PSNR={row['ours_raw_psnr']:.2f}  SSIM={row['ours_raw_ssim']:.4f}  ({row['ours_seconds']:.1f}s)")

            for a in cli.alphas:
                tag = str(a).replace('.', 'p')
                blend = ((1.0 - a) * low + a * den_raw).clamp(0, 1)
                row[f"ours_blend_alpha_a{tag}"] = a
                row[f"ours_blend_psnr_a{tag}"] = mod.compute_psnr(blend, full)
                row[f"ours_blend_ssim_a{tag}"] = mod.compute_ssim(blend, full)
                print(f"Ours a={a:g} PSNR={row[f'ours_blend_psnr_a{tag}']:.2f}  SSIM={row[f'ours_blend_ssim_a{tag}']:.4f}")

            if not cli.skip_f2n:
                print("Running F2N...")
                t0 = time.time()
                den_f2n = mod.run_f2n(low_train, args, device)
                den_f2n = mod.unpad_even(den_f2n, pad_hw)
                row["f2n_seconds"] = time.time() - t0
                row["f2n_psnr"] = mod.compute_psnr(den_f2n, full)
                row["f2n_ssim"] = mod.compute_ssim(den_f2n, full)
                print(f"F2N        PSNR={row['f2n_psnr']:.2f}  SSIM={row['f2n_ssim']:.4f}  ({row['f2n_seconds']:.1f}s)")

            if cli.save_images:
                stem = Path(str(key)).stem.replace("/", "_")
                mod.save_gray(low, os.path.join(cli.out_dir, f"{run_idx:04d}_{stem}_low.png"))
                mod.save_gray(full, os.path.join(cli.out_dir, f"{run_idx:04d}_{stem}_full.png"))
                mod.save_gray(den_raw, os.path.join(cli.out_dir, f"{run_idx:04d}_{stem}_ours_raw.png"))
                for a in cli.alphas:
                    tag = str(a).replace('.', 'p')
                    blend = ((1.0 - a) * low + a * den_raw).clamp(0, 1)
                    mod.save_gray(blend, os.path.join(cli.out_dir, f"{run_idx:04d}_{stem}_ours_a{tag}.png"))
                if not cli.skip_f2n:
                    mod.save_gray(den_f2n, os.path.join(cli.out_dir, f"{run_idx:04d}_{stem}_f2n.png"))

        except Exception as e:
            row["status"] = "error"
            row["error"] = repr(e)
            print(f"ERROR: {e}")

        rows.append(row)
        write_rows(cli.out_csv, rows)
        print(f"Wrote {cli.out_csv}")

    summarize(rows, cli.alphas)


if __name__ == "__main__":
    main()
