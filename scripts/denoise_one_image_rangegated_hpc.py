#!/usr/bin/env python3
"""
One-image test script for D45/sharp-kernel AdaptKPN vs F2N.

V4 adds a stronger F2N-style training loss for AdaptKPN: ELS + multi-scale L1 consistency + DoG edge preservation.

Default changes vs denoise_batches.py:
  1) Fixed PNG scaling instead of per-image min/max scaling.
  2) D45-friendly AdaptKPN presets.
  3) Poisson auxiliary loss disabled by default.
  4) Added two AdaptKPN loss modes: original ZS+Sobel and stronger F2N-style ELS+DoG.
  5) Optional F2N baseline with paper-like 500 epochs.
  6) Optional path auto-selection using root/dose/kernel/patient/index with robust QD/FD slice matching.
     For Mayo-style names with different QD/FD CT series IDs, default auto matching now falls back to CT instance number.

Example:
  python denoise_one_image_d45.py \
    --low "Preprocessed_512x512/512/Quarter Dose/1mm/Sharp Kernel (D45)/L096/0001.png" \
    --full "Preprocessed_512x512/512/Full Dose/1mm/Sharp Kernel (D45)/L096/0001.png" \
    --run-f2n --out-dir ./one_image_d45_test

Or select one matched slice automatically:
  python denoise_one_image_d45.py \
    --root "Preprocessed_512x512/512" \
    --kernel "Sharp Kernel (D45)" \
    --patient L096 --index 0 --run-f2n
"""

import argparse
import glob
import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Works both when filter2noise.py is beside this script and when using the original package layout.
try:
    from Filter2Noise.filter2noise import DenoisingPipeline, LossFunction, train_model
except Exception:
    try:
        from filter2noise import DenoisingPipeline, LossFunction, train_model
    except Exception:
        DenoisingPipeline = None
        LossFunction = None
        train_model = None


# -----------------------------
# AMP compatibility helpers
# -----------------------------
def make_grad_scaler(device: torch.device, amp: bool):
    """Return a GradScaler compatible with both old and new PyTorch."""
    enabled = bool(device.type == "cuda" and amp)

    # Newer PyTorch: torch.amp.GradScaler("cuda", enabled=...)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)

    # Older PyTorch: torch.cuda.amp.GradScaler(enabled=...)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    # Very old CPU-only fallback. It behaves like a disabled scaler.
    class _DummyScaler:
        def scale(self, loss):
            return loss
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
    return _DummyScaler()


def autocast_context(device: torch.device, amp: bool):
    """Return an autocast context compatible with both old and new PyTorch."""
    enabled = bool(device.type == "cuda" and amp)
    if not enabled:
        return nullcontext()

    # Newer PyTorch: torch.autocast(device_type=..., dtype=..., enabled=...)
    if hasattr(torch, "autocast"):
        try:
            return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
        except TypeError:
            pass

    # Older PyTorch: torch.cuda.amp.autocast(enabled=...)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)

    return nullcontext()


# -----------------------------
# Reproducibility and utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((x.detach().clamp(0, 1) - y.detach().clamp(0, 1)) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, eps))


def compute_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    x_np = x.detach().clamp(0, 1).cpu().numpy()[0, 0]
    y_np = y.detach().clamp(0, 1).cpu().numpy()[0, 0]
    return float(compare_ssim(x_np, y_np, data_range=1.0))


def load_ct_png_fixed(path: str, png_max: Optional[float] = None) -> torch.Tensor:
    """
    Load a PNG using fixed intensity scaling.

    Why: per-image min/max normalization changes the relative contrast of low-dose and
    full-dose images independently, which can distort PSNR/SSIM, especially for D45.

    png_max:
      None  -> infer from dtype: uint8=255, uint16=65535, else from actual global-ish range.
      value -> divide by this exact value, e.g. 255, 4095, 65535.
    """
    img = Image.open(path)
    arr_raw = np.asarray(img)

    if arr_raw.ndim == 3:
        arr_raw = arr_raw[..., 0]
    if arr_raw.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale PNG, got shape {arr_raw.shape} at {path}")

    arr = arr_raw.astype(np.float32)

    if png_max is None:
        if arr_raw.dtype == np.uint8:
            scale = 255.0
        elif arr_raw.dtype == np.uint16:
            scale = 65535.0
        else:
            # Fallback for float or uncommon PNG readers.
            # This is still fixed-style, not per-image min/max contrast stretching.
            max_val = float(np.nanmax(arr)) if arr.size else 1.0
            scale = 1.0 if max_val <= 1.0 else 255.0 if max_val <= 255.0 else 65535.0
    else:
        scale = float(png_max)

    if scale <= 0:
        raise ValueError(f"Invalid png_max={scale}")

    arr = np.clip(arr / scale, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


def maybe_center_crop(x: torch.Tensor, crop_size: Optional[int]) -> torch.Tensor:
    if crop_size is None or crop_size <= 0:
        return x.contiguous()

    _, _, h, w = x.shape
    crop = int(crop_size)
    if h < crop or w < crop:
        ph = max(crop - h, 0)
        pw = max(crop - w, 0)
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="reflect")
        _, _, h, w = x.shape

    top = (h - crop) // 2
    left = (w - crop) // 2
    return x[:, :, top:top + crop, left:left + crop].contiguous()


def pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad bottom/right if needed so ZS-N2N downsampler can use stride 2."""
    _, _, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x.contiguous(), (pad_h, pad_w)


def unpad_even(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h:
        x = x[..., :-pad_h, :]
    if pad_w:
        x = x[..., :, :-pad_w]
    return x.contiguous()


def save_gray(t: torch.Tensor, path: str) -> None:
    arr = (t.detach().clamp(0, 1).cpu().numpy()[0, 0] * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_comparison(low, den_a, full, out_dir, tag, den_b=None):
    imgs = [low.detach().clamp(0, 1).cpu().numpy()[0, 0], den_a.detach().clamp(0, 1).cpu().numpy()[0, 0]]
    titles = ["Noisy low-dose", "AdaptKPN-D45"]

    if den_b is not None:
        imgs.append(den_b.detach().clamp(0, 1).cpu().numpy()[0, 0])
        titles.append("F2N baseline")

    if full is not None:
        imgs.append(full.detach().clamp(0, 1).cpu().numpy()[0, 0])
        titles.append("Full-dose target")

    fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
    if len(imgs) == 1:
        axes = [axes]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{tag}_comparison.png")
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Path selection by robust slice matching
# -----------------------------
def _ct_suffix_key(path: str) -> Optional[str]:
    """
    Full suffix after '.CT.'. This works only when QD and FD share the same
    CT series/time identifiers. Your example does NOT share those identifiers,
    so this is not enough by itself.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    marker = ".CT."
    if marker not in name:
        return None
    return "CT." + name.split(marker, 1)[1]


def _ct_instance_key(path: str) -> Optional[str]:
    """
    Dose-invariant Mayo-style slice key.

    Your examples are:
      QD: L096_QD_1_SHARP_1.CT.0004.0001.2016....png
      FD: L096_FD_1_SHARP_1.CT.0006.0001.2016....png

    The CT series number differs (0004 vs 0006), and the timestamp differs, but
    the slice/instance token after the series number is shared (0001, 0002, ...).
    Therefore this function uses the second token after '.CT.' as the key.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    marker = ".CT."
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    parts = tail.split(".")
    if len(parts) < 2:
        return None
    # parts[0] is series/recon id; parts[1] is the common slice/instance number.
    return f"CT_INSTANCE_{parts[1]}"


def _dose_normalized_basename_key(path: str) -> str:
    """Normalize common QD/FD tokens while keeping the rest of the basename."""
    key = os.path.splitext(os.path.basename(path))[0]
    replacements = [
        ("_QD_", "_DOSE_"),
        ("_FD_", "_DOSE_"),
        ("_LD_", "_DOSE_"),
        ("_ND_", "_DOSE_"),
        ("Quarter Dose", "DOSE"),
        ("Full Dose", "DOSE"),
        ("quarter", "DOSE"),
        ("full", "DOSE"),
        ("low", "DOSE"),
    ]
    for old, new_token in replacements:
        key = key.replace(old, new_token)
    return key


def _make_unique_key_map(paths, key_fn):
    """Return {match_key: path}; duplicate or missing keys are omitted to avoid false matches."""
    grouped = {}
    missing = []
    for p in sorted(paths):
        k = key_fn(p)
        if k is None:
            missing.append(p)
            continue
        grouped.setdefault(k, []).append(p)
    unique = {k: v[0] for k, v in grouped.items() if len(v) == 1}
    dups = {k: v for k, v in grouped.items() if len(v) > 1}
    return unique, dups, missing


def _find_common_with_strategy(low_paths, full_paths, strategy: str):
    """Return low_map, full_map, common_keys, mode_name."""
    if strategy == "exact":
        low_map = {os.path.basename(p): p for p in low_paths}
        full_map = {os.path.basename(p): p for p in full_paths}
        common = sorted(low_map.keys() & full_map.keys())
        return low_map, full_map, common, "exact basename"

    if strategy == "dose-normalized":
        key_fn = _dose_normalized_basename_key
        mode = "dose-normalized basename"
    elif strategy == "ct-suffix":
        key_fn = _ct_suffix_key
        mode = "full CT suffix"
    elif strategy == "ct-instance":
        key_fn = _ct_instance_key
        mode = "CT instance number"
    else:
        raise ValueError(f"Unknown matching strategy: {strategy}")

    low_map, low_dups, low_missing = _make_unique_key_map(low_paths, key_fn)
    full_map, full_dups, full_missing = _make_unique_key_map(full_paths, key_fn)
    common = sorted(low_map.keys() & full_map.keys())

    if common and (low_dups or full_dups or low_missing or full_missing):
        print(
            f"  Warning for {mode} matching: "
            f"duplicate keys skipped low={len(low_dups)}, full={len(full_dups)}; "
            f"missing keys low={len(low_missing)}, full={len(full_missing)}"
        )

    return low_map, full_map, common, mode


def select_pair_from_dataset(
    root: str,
    dose: str,
    kernel: str,
    patient: str,
    index: int,
    match_mode: str = "auto",
    allow_sorted_fallback: bool = False,
) -> Tuple[str, str, str]:
    low_dir = os.path.join(root, dose, kernel, patient)
    full_dir = os.path.join(root, "Full Dose/1mm", kernel, patient)

    if not os.path.isdir(low_dir):
        raise FileNotFoundError(f"Missing low-dose directory: {low_dir}")
    if not os.path.isdir(full_dir):
        raise FileNotFoundError(f"Missing full-dose directory: {full_dir}")

    low_paths = sorted(glob.glob(os.path.join(low_dir, "*.png")))
    full_paths = sorted(glob.glob(os.path.join(full_dir, "*.png")))
    if not low_paths:
        raise RuntimeError(f"No PNG files found in low-dose directory: {low_dir}")
    if not full_paths:
        raise RuntimeError(f"No PNG files found in full-dose directory: {full_dir}")

    if match_mode == "auto":
        # Exact and full CT suffix are tried first. Your dataset will usually match
        # by ct-instance because QD/FD series ids and timestamps differ.
        strategies = ["exact", "dose-normalized", "ct-suffix", "ct-instance"]
    else:
        strategies = [match_mode]

    chosen = None
    tried_summaries = []
    for strategy in strategies:
        low_map, full_map, common, mode = _find_common_with_strategy(low_paths, full_paths, strategy)
        tried_summaries.append(f"{mode}: {len(common)} match(es)")
        if common:
            chosen = (low_map, full_map, common, mode)
            break

    if chosen is None and (allow_sorted_fallback or match_mode == "sorted-index"):
        # Last-resort option. This assumes sorted QD and sorted FD folders are aligned.
        # It is NOT enabled by default because it can hide pairing problems.
        n = min(len(low_paths), len(full_paths))
        common = [f"SORTED_INDEX_{i:04d}" for i in range(n)]
        low_map = {common[i]: low_paths[i] for i in range(n)}
        full_map = {common[i]: full_paths[i] for i in range(n)}
        chosen = (low_map, full_map, common, "sorted index fallback")
        tried_summaries.append(f"sorted index fallback: {len(common)} pair(s)")

    if chosen is None:
        low_examples = "\n    ".join(os.path.basename(p) for p in low_paths[:5])
        full_examples = "\n    ".join(os.path.basename(p) for p in full_paths[:5])
        low_instance_examples = sorted(filter(None, (_ct_instance_key(p) for p in low_paths[:5])))
        full_instance_examples = sorted(filter(None, (_ct_instance_key(p) for p in full_paths[:5])))
        raise RuntimeError(
            "No matched PNG pairs found.\n"
            f"Tried: {', '.join(tried_summaries)}\n"
            f"Low dir:  {low_dir}\n"
            f"Full dir: {full_dir}\n"
            f"Low PNG examples:\n    {low_examples}\n"
            f"Full PNG examples:\n    {full_examples}\n"
            f"Low CT-instance examples:  {low_instance_examples}\n"
            f"Full CT-instance examples: {full_instance_examples}\n"
            "Try --match-mode ct-instance, or bypass auto-matching with --low and --full. "
            "As a last resort only, use --allow-sorted-fallback."
        )

    low_map, full_map, common, mode = chosen

    if index < 0 or index >= len(common):
        raise IndexError(f"index={index} is outside matched range 0..{len(common)-1} using {mode} matching")

    key = common[index]
    print(f"  Tried matching: {', '.join(tried_summaries)}")
    print(f"  Matched {len(common)} pair(s) using {mode}; selected index={index}")
    print(f"  Match key: {key}")
    return low_map[key], full_map[key], f"{patient}_{key}"


# -----------------------------
# Approach A: D45-friendly AdaptKPN
# -----------------------------
class _KPNStage(nn.Module):
    """
    Range-gated adaptive kernel prediction stage.

    The original AdaptKPN predicts a positive spatial kernel and averages local
    patches. This version keeps the learned spatial kernel, but multiplies it by
    an intensity-similarity gate before normalization:

        w_final = softmax(KPN(x)) * exp(-(patch - center)^2 / (2*sigma_r^2))

    This is a KPN/bilateral hybrid: it is still a learned local kernel predictor,
    but it is constrained to avoid averaging across strong intensity jumps.
    """
    def __init__(
        self,
        n_chan: int,
        chan: int,
        k: int,
        smooth_mix: float,
        use_range_gate: bool = True,
        range_sigma_init: float = 0.05,
        range_sigma_min: float = 0.005,
        range_sigma_max: float = 0.25,
    ):
        super().__init__()
        self.k = int(k)
        self.pad = self.k // 2
        self.smooth_mix = float(smooth_mix)
        self.use_range_gate = bool(use_range_gate)
        self.range_sigma_min = float(range_sigma_min)
        self.range_sigma_max = float(range_sigma_max)

        self.net = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, 3, padding=1, groups=n_chan, bias=False),
            nn.Conv2d(n_chan, chan, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan, chan, 3, padding=1, groups=chan, bias=False),
            nn.Conv2d(chan, chan, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan, self.k * self.k, 1, bias=True),
        )

        # Learn one range sigma per stage. We parameterize it by log sigma so it
        # stays positive, then clamp to a defensible range during the forward pass.
        init = max(float(range_sigma_init), 1e-6)
        self.log_sigma_r = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))

        # Start close to identity. This matters for sharp kernels where too much
        # smoothing can reduce PSNR.
        with torch.no_grad():
            final_conv = self.net[-1]
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)
            final_conv.bias[(self.k * self.k) // 2] = 3.0

    def current_sigma_r(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_r).clamp(self.range_sigma_min, self.range_sigma_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        spatial_weights = F.softmax(self.net(x), dim=1)  # [B, K*K, H, W]
        patches = F.unfold(x, kernel_size=self.k, padding=self.pad)
        patches = patches.reshape(b, c, self.k * self.k, h, w)

        if self.use_range_gate:
            center = x.unsqueeze(2)  # [B, C, 1, H, W]
            sigma = self.current_sigma_r().to(dtype=x.dtype, device=x.device)
            # Mean across channels keeps the gate shape compatible with grayscale
            # and multi-channel inputs.
            diff2 = (patches - center).pow(2).mean(dim=1)  # [B, K*K, H, W]
            range_gate = torch.exp(-diff2 / (2.0 * sigma * sigma + 1e-12))
            weights = spatial_weights * range_gate
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
        else:
            weights = spatial_weights

        filtered = (patches * weights.unsqueeze(1)).sum(dim=2)
        return (self.smooth_mix * filtered + (1.0 - self.smooth_mix) * x).contiguous()


class AdaptKPN(nn.Module):
    def __init__(
        self,
        n_chan: int = 1,
        chan: int = 16,
        k: int = 5,
        stages: int = 3,
        smooth_mix: float = 0.75,
        use_range_gate: bool = True,
        range_sigma_init: float = 0.05,
        range_sigma_min: float = 0.005,
        range_sigma_max: float = 0.25,
    ):
        super().__init__()
        self.stages = nn.ModuleList([
            _KPNStage(
                n_chan=n_chan,
                chan=chan,
                k=k,
                smooth_mix=smooth_mix,
                use_range_gate=use_range_gate,
                range_sigma_init=range_sigma_init,
                range_sigma_min=range_sigma_min,
                range_sigma_max=range_sigma_max,
            )
            for _ in range(stages)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for stage in self.stages:
            out = stage(out)
        return out.contiguous()

    def sigma_r_values(self):
        vals = []
        for stage in self.stages:
            if hasattr(stage, "current_sigma_r"):
                vals.append(float(stage.current_sigma_r().detach().cpu().item()))
        return vals


# -----------------------------
# Self-supervised losses
# -----------------------------
def pair_downsampler(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0, f"Expected even spatial dims, got {h}x{w}"

    f1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    f2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    return (
        F.conv2d(img, f1, stride=2, groups=c).contiguous(),
        F.conv2d(img, f2, stride=2, groups=c).contiguous(),
    )



def ELS_local_shuffle(img: torch.Tensor) -> torch.Tensor:
    """
    Euclidean Local Shuffle used by F2N: within each 2x2 block, swap the closest-intensity pair.
    This decorrelates local CT noise while mostly preserving local anatomy statistics.
    """
    b, c, h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0, f"ELS needs even H/W, got {h}x{w}"

    blocks = img.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
    m = b * (h // 2) * (w // 2)
    flat = blocks.reshape(m, c, 2, 2).permute(0, 2, 3, 1).reshape(m, 4, c)

    diff = flat.unsqueeze(2) - flat.unsqueeze(1)
    dists = (diff ** 2).sum(dim=-1)
    eye = torch.eye(4, device=img.device, dtype=torch.bool).unsqueeze(0)
    dists = dists.masked_fill(eye, float("inf"))

    idx = torch.argmin(dists.reshape(m, -1), dim=1)
    p = idx // 4
    q = idx % 4
    out = flat.clone()
    rows = torch.arange(m, device=img.device)
    tmp = out[rows, p, :].clone()
    out[rows, p, :] = out[rows, q, :]
    out[rows, q, :] = tmp

    out = out.view(m, 2, 2, c).permute(0, 3, 1, 2)
    out = out.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return out.contiguous().view(b, c, h, w)


class F2NStyleLoss:
    """
    F2N-like loss for AdaptKPN.

    It mirrors the important parts of Filter2Noise's LossFunction:
      1) Downsample noisy image into two views.
      2) Apply ELS to the two views.
      3) Denoise the shuffled views.
      4) Enforce multi-scale consistency with denoised full-resolution output.
      5) Preserve edges with a weak Difference-of-Gaussians response.

    This is usually stronger for D45 than the earlier ZS-N2N+Sobel objective, which often
    converges near identity and gives only a small gain over the noisy input.
    """
    def __init__(self, device: torch.device, lambda_edge: float = 350.0, kernel_size: int = 7,
                 sigma_narrow: float = 9.0, sigma_wide: float = 10.0):
        self.device = device
        self.lambda_edge = float(lambda_edge)
        self.kernel_size = int(kernel_size)
        self.sigma_narrow = float(sigma_narrow)
        self.sigma_wide = float(sigma_wide)
        coords = torch.arange(self.kernel_size, dtype=torch.float32, device=device) - self.kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g1 = torch.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma_narrow ** 2))
        g1 = g1 / g1.sum()
        g2 = torch.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma_wide ** 2))
        g2 = g2 / g2.sum()
        self.base_dog = (g1 - g2).unsqueeze(0).unsqueeze(0)

    def __call__(self, noisy: torch.Tensor, model: nn.Module) -> torch.Tensor:
        c = noisy.shape[1]
        dog = self.base_dog.to(device=noisy.device, dtype=noisy.dtype).repeat(c, 1, 1, 1)

        y1, y2 = pair_downsampler(noisy)
        y1s = ELS_local_shuffle(y1)
        y2s = ELS_local_shuffle(y2)

        p1 = model(y1s).contiguous()
        p2 = model(y2s).contiguous()

        # Same weighting pattern as F2N's public implementation.
        loss_resolution = (1.0 / 3.0) * F.l1_loss(p1, p2)

        den_full = model(noisy).contiguous()
        d1, d2 = pair_downsampler(den_full)
        loss_cross_scale = (1.0 / 3.0) * (F.l1_loss(p1, d1) + F.l1_loss(p2, d2))
        loss_denoise = (1.0 / 3.0) * F.l1_loss(d1, d2)

        edges_noisy = F.conv2d(noisy, dog, padding=self.kernel_size // 2, groups=c)
        edges_den = F.conv2d(den_full, dog, padding=self.kernel_size // 2, groups=c)
        loss_edge = self.lambda_edge * F.l1_loss(torch.abs(edges_noisy), torch.abs(edges_den))

        return loss_resolution + loss_cross_scale + loss_denoise + loss_edge


def zs_n2n_loss(model: nn.Module, y: torch.Tensor) -> torch.Tensor:
    y1, y2 = pair_downsampler(y)
    p1 = model(y1).contiguous()
    p2 = model(y2).contiguous()

    # Symmetric cross-prediction + consistency with full-resolution prediction.
    loss_res = 0.5 * (F.mse_loss(y1, p2) + F.mse_loss(y2, p1))
    d1, d2 = pair_downsampler(model(y).contiguous())
    loss_con = 0.5 * (F.mse_loss(p1, d1) + F.mse_loss(p2, d2))
    return loss_res + loss_con


def poisson_n2n_loss_ct(model: nn.Module, noisy: torch.Tensor, peak: float, p: float) -> torch.Tensor:
    # Optional only. Off by default because reconstructed/windowed CT PNGs are not raw counts.
    counts = (noisy.clamp(0, 1) * peak).round().contiguous()
    c1 = torch.binomial(counts, torch.full_like(counts, p))
    c2 = counts - c1
    x1 = (c1 / (p * peak)).contiguous()
    x2 = (c2 / ((1.0 - p) * peak)).contiguous()
    return 0.5 * (F.mse_loss(model(x1), x2) + F.mse_loss(model(x2), x1))


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def edge_preserve_loss(noisy: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    # L1 is less aggressive than MSE and works well for preserving D45 high frequencies.
    return F.l1_loss(sobel_edges(denoised), sobel_edges(noisy))


@torch.no_grad()
def apply_transform(x: torch.Tensor, t: int) -> torch.Tensor:
    if t < 4:
        return torch.rot90(x, k=t, dims=(-2, -1)).contiguous()
    y = torch.rot90(x, k=t - 4, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)).contiguous()


@torch.no_grad()
def invert_transform(x: torch.Tensor, t: int) -> torch.Tensor:
    if t < 4:
        return torch.rot90(x, k=(4 - t) % 4, dims=(-2, -1)).contiguous()
    y = torch.flip(x, dims=(-1,))
    return torch.rot90(y, k=(4 - (t - 4)) % 4, dims=(-2, -1)).contiguous()


@torch.no_grad()
def denoise_tta(model: nn.Module, y: torch.Tensor, use_tta: bool) -> torch.Tensor:
    if not use_tta:
        return model(y).contiguous()
    acc = torch.zeros_like(y)
    for t in range(8):
        acc += invert_transform(model(apply_transform(y, t)).contiguous(), t)
    return (acc / 8.0).contiguous()



@torch.no_grad()
def oracle_blend_diagnostic(noisy: torch.Tensor, denoised: torch.Tensor, full: torch.Tensor) -> dict:
    """
    Diagnostic only, not a benchmark method: uses full-dose target to find the best linear
    blend between noisy and AdaptKPN output. If alpha clips at 1.0, the model is probably
    under-denoising. If alpha is near 0, it is probably over-smoothing.
    """
    n = noisy.detach().clamp(0, 1)
    d = denoised.detach().clamp(0, 1)
    f = full.detach().clamp(0, 1)
    direction = d - n
    denom = torch.sum(direction * direction).item()
    if denom <= 1e-12:
        alpha = 0.0
    else:
        alpha = torch.sum((f - n) * direction).item() / denom
    alpha01 = float(max(0.0, min(1.0, alpha)))
    blend = (n + alpha01 * direction).clamp(0, 1)
    return {
        "alpha_unclipped": float(alpha),
        "alpha_0_1": alpha01,
        "psnr": compute_psnr(blend, f),
        "ssim": compute_ssim(blend, f),
    }


def train_adaptkpn_d45(noisy: torch.Tensor, args, device: torch.device) -> Tuple[nn.Module, float]:
    set_seed(args.seed)

    model = AdaptKPN(
        n_chan=noisy.shape[1],
        chan=args.kpn_chan,
        k=args.kpn_k,
        stages=args.kpn_stages,
        smooth_mix=args.smooth_mix,
        use_range_gate=getattr(args, "use_range_gate", True),
        range_sigma_init=getattr(args, "range_sigma_init", 0.05),
        range_sigma_min=getattr(args, "range_sigma_min", 0.005),
        range_sigma_max=getattr(args, "range_sigma_max", 0.25),
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if args.optimizer == "adamw":
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "onecycle":
        sch = OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=1)
    else:
        sch = optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    f2n_style_loss = F2NStyleLoss(device, lambda_edge=args.adapt_lambda_edge) if args.loss_mode == "f2n" else None

    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.use_ema else None
    scaler = make_grad_scaler(device, args.amp)

    last_loss = 0.0
    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        with autocast_context(device, args.amp):
            if args.loss_mode == "f2n":
                loss = f2n_style_loss(noisy, model)
            else:
                den_full = model(noisy).contiguous()
                loss = zs_n2n_loss(model, noisy)

                if args.edge_weight > 0:
                    loss = loss + args.edge_weight * edge_preserve_loss(noisy, den_full)

            if args.poisson_weight > 0:
                loss = loss + args.poisson_weight * poisson_n2n_loss_ct(
                    model, noisy, peak=args.poisson_peak, p=args.poisson_thin_p
                )

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sch.step()

        last_loss = float(loss.detach().item())

        if ema is not None:
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema:
                    ema[k].mul_(args.ema_decay).add_(sd[k].detach(), alpha=1.0 - args.ema_decay)

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"  AdaptKPN epoch {epoch:4d}/{args.epochs} | loss={last_loss:.6f} | lr={opt.param_groups[0]['lr']:.2e}")

    if ema is not None:
        model.load_state_dict({k: v.to(device) for k, v in ema.items()})

    model.eval()
    return model, last_loss


def run_f2n(noisy: torch.Tensor, args, device: torch.device) -> torch.Tensor:
    if DenoisingPipeline is None or LossFunction is None or train_model is None:
        raise ImportError("Could not import F2N classes. Put filter2noise.py beside this script or keep the Filter2Noise package available.")

    set_seed(args.seed)
    model = DenoisingPipeline(num_stages=args.f2n_stages, patch_size=args.f2n_patch_size).to(device)
    optimizer = AdamW(model.parameters(), lr=args.f2n_lr, weight_decay=args.f2n_weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.f2n_lr, epochs=args.f2n_epochs, steps_per_epoch=1)
    loss_fn = LossFunction(device, lambda_=args.f2n_lambda_edge)

    train_model(
        model=model,
        noisy=noisy,
        loss_function=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.f2n_epochs,
    )

    model.eval()
    with torch.no_grad():
        return model(noisy).clamp(0, 1).contiguous()


def parse_args():
    parser = argparse.ArgumentParser(description="Run D45-friendly AdaptKPN V4 on one CT PNG and optionally compare with F2N.")

    # Direct path mode
    parser.add_argument("--low", type=str, default=None, help="Low-dose/noisy PNG path.")
    parser.add_argument("--full", type=str, default=None, help="Optional full-dose/clean PNG path for metrics.")

    # Dataset selection mode
    parser.add_argument("--root", type=str, default="Preprocessed_512x512/512", help="Dataset root used if --low is omitted.")
    parser.add_argument("--dose", type=str, default="Quarter Dose/1mm")
    parser.add_argument("--kernel", type=str, default="Sharp Kernel (D45)")
    parser.add_argument("--patient", type=str, default="L096")
    parser.add_argument("--index", type=int, default=0, help="0-based matched slice index used if --low is omitted.")
    parser.add_argument("--match-mode", type=str, default="auto",
                        choices=["auto", "exact", "dose-normalized", "ct-suffix", "ct-instance", "sorted-index"],
                        help="How to match low/full PNGs. Default auto tries exact, dose-normalized, CT suffix, then CT instance number.")
    parser.add_argument("--allow-sorted-fallback", action="store_true",
                        help="Last-resort pairing by sorted list position if all key-based matching fails. Use only if you know folders are aligned.")

    # IO and preprocessing
    parser.add_argument("--out-dir", type=str, default="./one_image_d45_test")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--png-max", type=float, default=None, help="Fixed PNG divisor. Defaults: uint8=255, uint16=65535.")
    parser.add_argument("--crop-size", type=int, default=0, help="Optional center crop. 0 means full image.")

    # General training
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True, help="Use CUDA mixed precision. Enabled by default.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile. Usually slower for a quick one-image test startup.")

    # AdaptKPN D45 defaults
    parser.add_argument("--epochs", type=int, default=500, help="D45 V4 default. Try 300/500/800.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "step"])
    parser.add_argument("--step-size", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--loss-mode", type=str, default="f2n", choices=["f2n", "zs_sobel"],
                        help="AdaptKPN training loss. V4 default f2n uses ELS+multi-scale+DoG; zs_sobel is the V3 loss.")
    parser.add_argument("--adapt-lambda-edge", type=float, default=350.0,
                        help="DoG edge weight for --loss-mode f2n. Sweep 200,300,350,450,500.")
    parser.add_argument("--kpn-chan", type=int, default=16)
    parser.add_argument("--kpn-k", type=int, default=5)
    parser.add_argument("--kpn-stages", type=int, default=3)
    parser.add_argument("--smooth-mix", type=float, default=0.75, help="Filtered/output mix. Original was 0.8; V4 D45 default is 0.75.")
    parser.add_argument("--range-gate", action="store_true", default=True, help="Enable bilateral/intensity range gate inside AdaptKPN. Enabled by default.")
    parser.add_argument("--no-range-gate", dest="range_gate", action="store_false")
    parser.add_argument("--range-sigma-init", type=float, default=0.05, help="Initial learnable range sigma in normalized [0,1] units.")
    parser.add_argument("--range-sigma-min", type=float, default=0.005)
    parser.add_argument("--range-sigma-max", type=float, default=0.25)
    parser.add_argument("--edge-weight", type=float, default=0.02, help="Only used by --loss-mode zs_sobel. Sweep 0.005,0.01,0.02,0.05.")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--self-ensemble", action="store_true", help="Off by default for D45; enable as an ablation.")

    # Poisson optional. Off by default for D45.
    parser.add_argument("--poisson-weight", type=float, default=0.0, help="Off by default. Gentle test: 0.05 with --poisson-peak 1000.")
    parser.add_argument("--poisson-peak", type=float, default=1000.0)
    parser.add_argument("--poisson-thin-p", type=float, default=0.5)

    # F2N baseline optional
    parser.add_argument("--run-f2n", action="store_true", help="Also run F2N baseline on the same image.")
    parser.add_argument("--f2n-epochs", type=int, default=500)
    parser.add_argument("--f2n-stages", type=int, default=2)
    parser.add_argument("--f2n-patch-size", type=int, default=8)
    parser.add_argument("--f2n-lr", type=float, default=1e-3)
    parser.add_argument("--f2n-weight-decay", type=float, default=0.01)
    parser.add_argument("--f2n-lambda-edge", type=float, default=350.0)

    parser.add_argument("--report-oracle-blend", action="store_true", default=True,
                        help="Diagnostic only when --full is present: reports best low/denoised blend using full target.")
    parser.add_argument("--no-report-oracle-blend", dest="report_oracle_blend", action="store_false")
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    # Keep imported-runner and CLI attribute names compatible.
    if not hasattr(args, "use_range_gate"):
        args.use_range_gate = getattr(args, "range_gate", True)
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.low is None:
        low_path, full_path, sample_tag = select_pair_from_dataset(
            root=args.root,
            dose=args.dose,
            kernel=args.kernel,
            patient=args.patient,
            index=args.index,
            match_mode=args.match_mode,
            allow_sorted_fallback=args.allow_sorted_fallback,
        )
        args.low = low_path
        args.full = full_path
        if args.tag is None:
            args.tag = sample_tag.replace("/", "_").replace(" ", "_")
    else:
        if args.tag is None:
            args.tag = os.path.splitext(os.path.basename(args.low))[0]

    print("\nSelected image")
    print(f"  low : {args.low}")
    print(f"  full: {args.full if args.full else '(not provided)'}")
    print(f"  device: {device}")
    print("\nAdaptKPN-D45 V4 config")
    print(f"  loss_mode={args.loss_mode}, epochs={args.epochs}, optimizer={args.optimizer}, scheduler={args.scheduler}")
    print(f"  k={args.kpn_k}, stages={args.kpn_stages}, chan={args.kpn_chan}, smooth_mix={args.smooth_mix}")
    print(f"  adapt_lambda_edge={args.adapt_lambda_edge}, edge_weight={args.edge_weight}, poisson_weight={args.poisson_weight}, self_ensemble={args.self_ensemble}")
    print(f"  fixed png scaling divisor: {args.png_max if args.png_max is not None else 'dtype inferred'}")

    low = load_ct_png_fixed(args.low, png_max=args.png_max).to(device)
    full = load_ct_png_fixed(args.full, png_max=args.png_max).to(device) if args.full else None

    crop_size = None if args.crop_size <= 0 else args.crop_size
    low = maybe_center_crop(low, crop_size)
    if full is not None:
        full = maybe_center_crop(full, crop_size)
        if low.shape != full.shape:
            raise ValueError(f"Low/full shapes differ after loading/crop: {tuple(low.shape)} vs {tuple(full.shape)}")

    # Pad only for training/inference if odd dimensions; unpad output before saving/metrics.
    low_train, pad_hw = pad_to_even(low)

    metrics = {}
    if full is not None:
        metrics["noisy_psnr"] = compute_psnr(low, full)
        metrics["noisy_ssim"] = compute_ssim(low, full)
        print(f"\nNoisy input       PSNR={metrics['noisy_psnr']:.2f} dB  SSIM={metrics['noisy_ssim']:.4f}")

    # Approach A
    t0 = time.time()
    model_a, final_loss = train_adaptkpn_d45(low_train, args, device)
    with torch.no_grad():
        den_a = denoise_tta(model_a, low_train, use_tta=args.self_ensemble).clamp(0, 1).contiguous()
    den_a = unpad_even(den_a, pad_hw)
    t_a = time.time() - t0

    metrics["adaptkpn_seconds"] = t_a
    metrics["adaptkpn_final_loss"] = final_loss
    if full is not None:
        metrics["adaptkpn_psnr"] = compute_psnr(den_a, full)
        metrics["adaptkpn_ssim"] = compute_ssim(den_a, full)
        print(f"AdaptKPN-D45 V4   PSNR={metrics['adaptkpn_psnr']:.2f} dB  SSIM={metrics['adaptkpn_ssim']:.4f}  ({t_a:.1f}s)")
        if args.report_oracle_blend:
            diag = oracle_blend_diagnostic(low, den_a, full)
            metrics["adaptkpn_oracle_blend"] = diag
            print(
                "  diagnostic blend with full target: "
                f"alpha={diag['alpha_unclipped']:.2f} (clipped {diag['alpha_0_1']:.2f}), "
                f"PSNR={diag['psnr']:.2f} dB, SSIM={diag['ssim']:.4f}"
            )
    else:
        print(f"AdaptKPN-D45 V4 done ({t_a:.1f}s)")

    # Optional F2N baseline
    den_b = None
    if args.run_f2n:
        print("\nRunning F2N baseline...")
        t0 = time.time()
        den_b = run_f2n(low_train, args, device)
        den_b = unpad_even(den_b, pad_hw)
        t_b = time.time() - t0
        metrics["f2n_seconds"] = t_b
        if full is not None:
            metrics["f2n_psnr"] = compute_psnr(den_b, full)
            metrics["f2n_ssim"] = compute_ssim(den_b, full)
            print(f"F2N baseline      PSNR={metrics['f2n_psnr']:.2f} dB  SSIM={metrics['f2n_ssim']:.4f}  ({t_b:.1f}s)")
        else:
            print(f"F2N baseline done ({t_b:.1f}s)")

    # Save outputs
    safe_tag = args.tag.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    low_out = os.path.join(args.out_dir, f"{safe_tag}_low.png")
    den_a_out = os.path.join(args.out_dir, f"{safe_tag}_adaptkpn_d45.png")
    save_gray(low, low_out)
    save_gray(den_a, den_a_out)

    if full is not None:
        save_gray(full, os.path.join(args.out_dir, f"{safe_tag}_full.png"))
    if den_b is not None:
        save_gray(den_b, os.path.join(args.out_dir, f"{safe_tag}_f2n.png"))

    comp_path = save_comparison(low, den_a, full, args.out_dir, safe_tag, den_b=den_b)

    report = {
        "low_path": args.low,
        "full_path": args.full,
        "device": str(device),
        "config": {
            "loss_mode": args.loss_mode,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "adapt_lambda_edge": args.adapt_lambda_edge,
            "kpn_k": args.kpn_k,
            "kpn_stages": args.kpn_stages,
            "smooth_mix": args.smooth_mix,
            "edge_weight": args.edge_weight,
            "poisson_weight": args.poisson_weight,
            "poisson_peak": args.poisson_peak,
            "self_ensemble": args.self_ensemble,
            "png_max": args.png_max,
            "crop_size": crop_size,
            "f2n_epochs": args.f2n_epochs if args.run_f2n else None,
        },
        "metrics": metrics,
        "outputs": {
            "low": low_out,
            "adaptkpn_d45": den_a_out,
            "comparison": comp_path,
        },
    }
    if den_b is not None:
        report["outputs"]["f2n"] = os.path.join(args.out_dir, f"{safe_tag}_f2n.png")

    report_path = os.path.join(args.out_dir, f"{safe_tag}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved")
    print(f"  denoised:   {den_a_out}")
    print(f"  comparison: {comp_path}")
    print(f"  report:     {report_path}")

    if full is not None:
        noisy = metrics.get("noisy_psnr")
        ours = metrics.get("adaptkpn_psnr")
        f2n = metrics.get("f2n_psnr")
        print("\nQuick check")
        print(f"  AdaptKPN gain over noisy: {ours - noisy:+.2f} dB")
        if "adaptkpn_oracle_blend" in metrics:
            ob = metrics["adaptkpn_oracle_blend"]
            print(f"  Oracle blend alpha:       {ob['alpha_unclipped']:.2f} (diagnostic only)")
        if f2n is not None:
            print(f"  AdaptKPN - F2N:          {ours - f2n:+.2f} dB")


if __name__ == "__main__":
    main()
