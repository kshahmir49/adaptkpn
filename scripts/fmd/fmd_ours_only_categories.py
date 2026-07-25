#!/usr/bin/env python3
"""
Run ONLY our Range-Gated AdaptKPN on paired FMD/FMD-like microscopy images.

Expected structure:
  <root>/raw/<name>.png
  <root>/gt/<name>.png

Example names:
  Confocal_BPAE_B_1.png
  TwoPhoton_MICE_G_3.png
  TwoPhoton_BPAE_R_12.png

Default categories:
  TwoPhoton_MICE
  TwoPhoton_BPAE
  Confocal_BPAE

Outputs:
  - per-image CSV
  - per-category summary CSV
  - overall summary CSV
  - optional denoised images

GT is used only for PSNR/SSIM evaluation, never for training.
"""

import argparse
import csv
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    from zero_shot_denoising_core import (
        select_device,
        set_seed,
        ensure_dir,
        compute_psnr,
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
    h, w = x.shape[:2]
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    return x[:h2, :w2, ...]


def load_image(path: Path, channels: str, crop: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path)
    if channels == "gray":
        img = img.convert("L")
        arr = np.asarray(img)
        arr = normalize_array(arr)
        arr = center_crop_np(arr, crop)
        arr = make_even_np(arr)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()
    elif channels == "rgb":
        img = img.convert("RGB")
        arr = np.asarray(img)
        arr = normalize_array(arr)
        arr = center_crop_np(arr, crop)
        arr = make_even_np(arr)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    else:
        raise ValueError("channels must be 'gray' or 'rgb'")
    return tensor.to(device=device, dtype=torch.float32)


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


def parse_fmd_name(stem: str) -> Tuple[str, str, str]:
    """Return (category, channel, sample_id) from FMD-like stem.

    Confocal_BPAE_B_1 -> (Confocal_BPAE, B, 1)
    TwoPhoton_MICE_G_3 -> (TwoPhoton_MICE, G, 3)
    """
    parts = stem.split("_")
    if len(parts) >= 2:
        category = f"{parts[0]}_{parts[1]}"
    else:
        category = stem
    channel = parts[2] if len(parts) >= 3 else ""
    sample_id = parts[-1] if len(parts) >= 4 and parts[-1].isdigit() else ""
    return category, channel, sample_id


def list_pairs(root: Path, raw_subdir: str, gt_subdir: str, exts: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
    raw_dir = root / raw_subdir
    gt_dir = root / gt_subdir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Missing GT directory: {gt_dir}")

    raw_files: List[Path] = []
    for ext in exts:
        raw_files.extend(raw_dir.rglob(f"*{ext}"))
    raw_files = sorted(raw_files)

    pairs: List[Tuple[Path, Path]] = []
    for raw_path in raw_files:
        rel = raw_path.relative_to(raw_dir)
        gt_path = gt_dir / rel
        if gt_path.exists():
            pairs.append((raw_path, gt_path))
            continue

        # Fallback: same stem, any supported extension.
        hit = None
        for ext in exts:
            cand = gt_dir / rel.parent / f"{raw_path.stem}{ext}"
            if cand.exists():
                hit = cand
                break
        if hit is not None:
            pairs.append((raw_path, hit))
        else:
            print(f"Warning: no GT match for {raw_path}")

    if not pairs:
        raise RuntimeError(f"No raw/gt pairs found in {raw_dir} and {gt_dir}")
    return pairs


def filter_categories(pairs: List[Tuple[Path, Path]], categories: List[str], max_per_category: int) -> List[Tuple[Path, Path]]:
    cats = set(categories)
    counts: Dict[str, int] = {c: 0 for c in categories}
    selected: List[Tuple[Path, Path]] = []
    for raw_path, gt_path in pairs:
        category, _, _ = parse_fmd_name(raw_path.stem)
        if category not in cats:
            continue
        if max_per_category > 0 and counts[category] >= max_per_category:
            continue
        selected.append((raw_path, gt_path))
        counts[category] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Folder containing raw/ and gt/ subfolders")
    parser.add_argument("--raw-subdir", type=str, default="raw")
    parser.add_argument("--gt-subdir", type=str, default="gt")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["TwoPhoton_MICE", "TwoPhoton_BPAE", "Confocal_BPAE"],
        help="FMD categories to evaluate",
    )
    parser.add_argument("--max-per-category", type=int, default=0, help="0 = use all images in each category")
    parser.add_argument("--channels", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--crop", type=int, default=256, help="Center crop size. 0 = full image, cropped only to even H/W")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="fmd_ours_only_results")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--save-images", action="store_true")

    # Ours: frozen range-gated configuration.
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
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "fmd_ours_only_results.csv"
    by_category_csv = out_dir / "fmd_ours_only_by_category.csv"
    overall_csv = out_dir / "fmd_ours_only_overall.csv"

    device = select_device(args.device)
    all_pairs = list_pairs(root, args.raw_subdir, args.gt_subdir, exts=(".png", ".tif", ".tiff"))
    pairs = filter_categories(all_pairs, args.categories, args.max_per_category)

    print(f"Device: {device}")
    print(f"Root: {root}")
    print(f"Total pairs found: {len(all_pairs)}")
    print(f"Selected pairs: {len(pairs)}")
    print(f"Categories: {args.categories}")
    print(f"Channels: {args.channels}, crop={args.crop}")
    print(f"Ours config: sigma={args.range_sigma_init}, lambda_edge={args.lambda_edge}, epochs={args.epochs_ours}")
    print(f"Output CSV: {out_csv}")

    if len(pairs) == 0:
        found = sorted({parse_fmd_name(p[0].stem)[0] for p in all_pairs})
        raise RuntimeError(f"No pairs matched requested categories. Categories found: {found}")

    rows = []
    for idx, (raw_path, gt_path) in enumerate(pairs, start=1):
        category, channel, sample_id = parse_fmd_name(raw_path.stem)
        set_seed(args.seed + idx)

        noisy = load_image(raw_path, channels=args.channels, crop=args.crop, device=device)
        clean = load_image(gt_path, channels=args.channels, crop=args.crop, device=device)
        if noisy.shape != clean.shape:
            raise ValueError(f"Shape mismatch for {raw_path.name}: raw={tuple(noisy.shape)}, gt={tuple(clean.shape)}")

        row = {
            "image": raw_path.name,
            "category": category,
            "channel": channel,
            "sample_id": sample_id,
            "channels_mode": args.channels,
            "crop": args.crop,
            "height": int(noisy.shape[-2]),
            "width": int(noisy.shape[-1]),
            "seed": args.seed + idx,
            "device": str(device),
            "noisy_psnr": compute_psnr(noisy, clean),
            "noisy_ssim": compute_ssim_any(noisy, clean),
        }
        print(f"\n[{idx}/{len(pairs)}] {raw_path.name} [{category}] noisy={row['noisy_psnr']:.2f}/{row['noisy_ssim']:.4f}")

        den, elapsed, n_params = train_ours(noisy, args, seed=args.seed + 20000 + idx)
        row.update({
            "ours_psnr": compute_psnr(den, clean),
            "ours_ssim": compute_ssim_any(den, clean),
            "ours_time_s": elapsed,
            "ours_params": n_params,
            "delta_psnr": compute_psnr(den, clean) - row["noisy_psnr"],
            "delta_ssim": compute_ssim_any(den, clean) - row["noisy_ssim"],
        })
        print(f"  Ours={row['ours_psnr']:.2f}/{row['ours_ssim']:.4f}, gain={row['delta_psnr']:+.2f}/{row['delta_ssim']:+.4f}, time={elapsed:.1f}s")

        if args.save_images:
            save_tensor_image(noisy, out_dir / f"{raw_path.stem}_raw.png")
            save_tensor_image(den, out_dir / f"{raw_path.stem}_ours.png")
            save_tensor_image(clean, out_dir / f"{raw_path.stem}_gt.png")

        rows.append(row)
        # Write per-image results after every image, so partial runs are usable.
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Summaries.
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        metric_cols = ["noisy_psnr", "ours_psnr", "delta_psnr", "noisy_ssim", "ours_ssim", "delta_ssim", "ours_time_s", "ours_params"]

        by_cat = df.groupby("category", as_index=False)[metric_cols].mean(numeric_only=True)
        counts = df.groupby("category", as_index=False).size().rename(columns={"size": "n"})
        by_cat = counts.merge(by_cat, on="category")

        overall = pd.DataFrame([{"category": "OVERALL", "n": len(df), **df[metric_cols].mean(numeric_only=True).to_dict()}])

        by_cat.to_csv(by_category_csv, index=False)
        overall.to_csv(overall_csv, index=False)

        print("\n==================== PER-CATEGORY SUMMARY ====================")
        print(by_cat.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n====================== OVERALL SUMMARY ========================")
        print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved per-image: {out_csv}")
        print(f"Saved by category: {by_category_csv}")
        print(f"Saved overall: {overall_csv}")
    except Exception as e:
        print("Summary failed:", repr(e))


if __name__ == "__main__":
    main()
