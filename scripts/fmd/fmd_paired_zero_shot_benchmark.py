#!/usr/bin/env python3
"""
Paired real-noise FMD/FMD-like microscopy benchmark.

Expected structure:
  <root>/raw/<name>.png
  <root>/gt/<name>.png

Example:
  text_mix/raw/Confocal_BPAE_B_1.png
  text_mix/gt/Confocal_BPAE_B_1.png

Compares:
  1) Noisy raw input
  2) ZS-N2N baseline
  3) Range-Gated AdaptKPN (ours)

GT is used only for PSNR/SSIM evaluation, never for training.
"""

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

# Reuse the exact implementations used in the Kodak experiments.
try:
    from zero_shot_denoising_core import (
        select_device,
        set_seed,
        ensure_dir,
        compute_psnr,
        train_zsn2n,
        train_ours,
    )
except Exception as e:
    raise ImportError(
        "Could not import zero_shot_denoising_core.py. Put this script in the same folder "
        "as zero_shot_denoising_core.py. Original error: " + repr(e)
    )


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize image array to [0,1] using dtype-based scaling."""
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    else:
        out = arr.astype(np.float32)
        # If float-like image is not already [0,1], use a conservative max scaling.
        mx = float(np.nanmax(out)) if out.size else 1.0
        if mx > 1.5:
            out = out / mx
    return np.clip(out, 0.0, 1.0)


def center_crop_np(x: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return x
    h, w = x.shape[:2]
    if h < crop or w < crop:
        raise ValueError(f"Image is smaller than crop={crop}: shape={x.shape}")
    top = (h - crop) // 2
    left = (w - crop) // 2
    return x[top:top + crop, left:left + crop, ...]


def make_even_np(x: np.ndarray) -> np.ndarray:
    """Crop bottom/right by one pixel if needed so downsampler has even H,W."""
    h, w = x.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    return x[:h2, :w2, ...]


def load_microscopy_image(path: Path, channels: str, crop: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path)
    if channels == "gray":
        img = img.convert("L")
        arr = np.asarray(img)
        arr = normalize_array(arr)
        arr = center_crop_np(arr, crop)
        arr = make_even_np(arr)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()
    elif channels == "rgb":
        img = img.convert("RGB")
        arr = np.asarray(img)
        arr = normalize_array(arr)
        arr = center_crop_np(arr, crop)
        arr = make_even_np(arr)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    else:
        raise ValueError("channels must be 'gray' or 'rgb'")
    return t.to(device=device, dtype=torch.float32)


def tensor_to_np_for_ssim(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0, 1).cpu()[0]
    if x.shape[0] == 1:
        return x[0].numpy()
    return x.permute(1, 2, 0).numpy()


def compute_ssim_any(pred: torch.Tensor, target: torch.Tensor) -> float:
    if skimage_ssim is None:
        return float("nan")
    p = tensor_to_np_for_ssim(pred)
    t = tensor_to_np_for_ssim(target)
    if p.ndim == 2:
        return float(skimage_ssim(t, p, data_range=1.0))
    return float(skimage_ssim(t, p, data_range=1.0, channel_axis=-1))


def save_tensor_image(x: torch.Tensor, path: Path) -> None:
    arr = x.detach().clamp(0, 1).cpu()[0]
    if arr.shape[0] == 1:
        im = (arr[0].numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(im, mode="L").save(path)
    else:
        im = (arr.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(im, mode="RGB").save(path)


def parse_category(stem: str) -> Tuple[str, str]:
    # Confocal_BPAE_B_1 -> category=Confocal_BPAE_B, sample=1
    m = re.match(r"(.+)_([0-9]+)$", stem)
    if m:
        return m.group(1), m.group(2)
    return stem, ""


def list_pairs(root: Path, raw_subdir: str, gt_subdir: str, exts: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
    raw_dir = root / raw_subdir
    gt_dir = root / gt_subdir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Missing gt directory: {gt_dir}")

    raw_files = []
    for ext in exts:
        raw_files.extend(raw_dir.rglob(f"*{ext}"))
    raw_files = sorted(raw_files)

    pairs = []
    for rp in raw_files:
        rel = rp.relative_to(raw_dir)
        gp = gt_dir / rel
        if gp.exists():
            pairs.append((rp, gp))
        else:
            # Try same stem with any supported extension.
            candidates = []
            for ext in exts:
                candidates.append((gt_dir / rel.parent / (rp.stem + ext)))
            hit = next((c for c in candidates if c.exists()), None)
            if hit is not None:
                pairs.append((rp, hit))
            else:
                print(f"Warning: no GT match for {rp}")
    if not pairs:
        raise RuntimeError(f"No raw/gt pairs found under {raw_dir} and {gt_dir}")
    return pairs


def select_pairs(pairs: List[Tuple[Path, Path]], images: List[str]) -> List[Tuple[Path, Path]]:
    if len(images) == 1 and images[0].lower() == "all":
        return pairs
    selected = []
    by_name = {p[0].name: p for p in pairs}
    by_stem = {p[0].stem: p for p in pairs}
    for token in images:
        if token.isdigit():
            idx = int(token) - 1
            if idx < 0 or idx >= len(pairs):
                raise IndexError(f"Image index {token} out of range 1..{len(pairs)}")
            selected.append(pairs[idx])
        elif token in by_name:
            selected.append(by_name[token])
        elif token in by_stem:
            selected.append(by_stem[token])
        else:
            raise KeyError(f"Could not find image '{token}' by index, filename, or stem")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Folder containing raw/ and gt/")
    parser.add_argument("--raw-subdir", type=str, default="raw")
    parser.add_argument("--gt-subdir", type=str, default="gt")
    parser.add_argument("--images", nargs="+", default=["all"], help="all, 1 2 3, or filenames/stems")
    parser.add_argument("--channels", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--crop", type=int, default=0, help="Center crop size. 0 = use full image, cropped only to even H/W.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="fmd_results")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--save-images", action="store_true")

    # ZS-N2N defaults from notebook/Kodak script.
    parser.add_argument("--run-zsn2n", action="store_true", default=True)
    parser.add_argument("--no-zsn2n", action="store_false", dest="run_zsn2n")
    parser.add_argument("--epochs-zsn2n", type=int, default=2000)
    parser.add_argument("--lr-zsn2n", type=float, default=1e-3)
    parser.add_argument("--zsn2n-step-size", type=int, default=1500)
    parser.add_argument("--zsn2n-gamma", type=float, default=0.5)

    # Ours.
    parser.add_argument("--run-ours", action="store_true", default=True)
    parser.add_argument("--no-ours", action="store_false", dest="run_ours")
    parser.add_argument("--epochs-ours", type=int, default=500)
    parser.add_argument("--lr-ours", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-edge", type=float, default=350.0)
    parser.add_argument("--kpn-chan", type=int, default=16)
    parser.add_argument("--kpn-k", type=int, default=5)
    parser.add_argument("--kpn-stages", type=int, default=3)
    parser.add_argument("--smooth-mix", type=float, default=0.75)
    parser.add_argument("--range-sigma-init", type=float, default=0.08)

    args = parser.parse_args()
    root = Path(args.root)
    device = select_device(args.device)
    ensure_dir(args.out_dir)
    out_csv = Path(args.out_csv) if args.out_csv else Path(args.out_dir) / "fmd_paired_results.csv"

    pairs_all = list_pairs(root, args.raw_subdir, args.gt_subdir, exts=(".png", ".tif", ".tiff"))
    pairs = select_pairs(pairs_all, args.images)

    print(f"Device: {device}")
    print(f"Root: {root}")
    print(f"Total pairs found: {len(pairs_all)}")
    print(f"Pairs selected: {len(pairs)}")
    print(f"Channels: {args.channels}, crop={args.crop}")
    print(f"Output CSV: {out_csv}")

    rows = []
    for i, (raw_path, gt_path) in enumerate(pairs, start=1):
        set_seed(args.seed + i)
        noisy = load_microscopy_image(raw_path, channels=args.channels, crop=args.crop, device=device)
        clean = load_microscopy_image(gt_path, channels=args.channels, crop=args.crop, device=device)
        if noisy.shape != clean.shape:
            raise ValueError(f"Shape mismatch for {raw_path.name}: raw {tuple(noisy.shape)}, gt {tuple(clean.shape)}")

        category, sample_id = parse_category(raw_path.stem)
        row = {
            "image": raw_path.name,
            "category": category,
            "sample_id": sample_id,
            "channels": args.channels,
            "crop": args.crop,
            "seed": args.seed + i,
            "device": str(device),
            "height": int(noisy.shape[-2]),
            "width": int(noisy.shape[-1]),
            "noisy_psnr": compute_psnr(noisy, clean),
            "noisy_ssim": compute_ssim_any(noisy, clean),
        }
        print(f"\n[{i}/{len(pairs)}] {raw_path.name}: noisy PSNR={row['noisy_psnr']:.2f}, SSIM={row['noisy_ssim']:.4f}")

        if args.save_images:
            prefix = Path(args.out_dir) / raw_path.stem
            save_tensor_image(clean, prefix.with_name(prefix.name + "_gt.png"))
            save_tensor_image(noisy, prefix.with_name(prefix.name + "_raw.png"))

        if args.run_zsn2n:
            den, elapsed, n_params = train_zsn2n(
                noisy,
                epochs=args.epochs_zsn2n,
                lr=args.lr_zsn2n,
                step_size=args.zsn2n_step_size,
                gamma=args.zsn2n_gamma,
                seed=args.seed + 10000 + i,
            )
            row.update({
                "zsn2n_psnr": compute_psnr(den, clean),
                "zsn2n_ssim": compute_ssim_any(den, clean),
                "zsn2n_time_s": elapsed,
                "zsn2n_params": n_params,
            })
            print(f"  ZS-N2N: PSNR={row['zsn2n_psnr']:.2f}, SSIM={row['zsn2n_ssim']:.4f}, time={elapsed:.1f}s")
            if args.save_images:
                save_tensor_image(den, Path(args.out_dir) / f"{raw_path.stem}_zsn2n.png")

        if args.run_ours:
            den, elapsed, n_params = train_ours(noisy, args, seed=args.seed + 20000 + i)
            row.update({
                "ours_psnr": compute_psnr(den, clean),
                "ours_ssim": compute_ssim_any(den, clean),
                "ours_time_s": elapsed,
                "ours_params": n_params,
            })
            print(f"  Ours:   PSNR={row['ours_psnr']:.2f}, SSIM={row['ours_ssim']:.4f}, time={elapsed:.1f}s")
            if args.save_images:
                save_tensor_image(den, Path(args.out_dir) / f"{raw_path.stem}_ours.png")

        rows.append(row)
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        print("\nOverall summary:")
        cols = [c for c in ["noisy_psnr", "zsn2n_psnr", "ours_psnr", "noisy_ssim", "zsn2n_ssim", "ours_ssim", "zsn2n_time_s", "ours_time_s"] if c in df.columns]
        print(df[cols].mean(numeric_only=True).to_string())
        if "category" in df.columns:
            print("\nBy category:")
            print(df.groupby("category")[cols].mean(numeric_only=True).to_string())
    except Exception as e:
        print("Summary failed:", repr(e))


if __name__ == "__main__":
    main()
