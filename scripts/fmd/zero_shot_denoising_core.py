#!/usr/bin/env python3
"""
Shared zero-shot denoising utilities.

This module contains the reusable code used by the FMD microscopy scripts:
  - device and seed utilities
  - PSNR and SSIM metrics
  - ZS-N2N baseline
  - range-gated adaptive KPN model
  - F2N-style self-supervised loss
  - training functions for ZS-N2N and our method

Natural-image benchmark code has been intentionally removed.
"""

import math
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def tensor_to_numpy_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
    return x


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, eps))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    if skimage_ssim is None:
        return float("nan")
    p = tensor_to_numpy_img(pred)
    t = tensor_to_numpy_img(target)
    return float(skimage_ssim(t, p, data_range=1.0, channel_axis=-1))


def pair_downsampler(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = img.shape[1]
    f1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
    f2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], dtype=img.dtype, device=img.device).repeat(c, 1, 1, 1)
    return (
        F.conv2d(img, f1, stride=2, groups=c).contiguous(),
        F.conv2d(img, f2, stride=2, groups=c).contiguous(),
    )


def els(img: torch.Tensor) -> torch.Tensor:
    """Euclidean Local Shuffle for RGB/grayscale tensors [B,C,H,W]."""
    B, C, H, W = img.shape
    assert H % 2 == 0 and W % 2 == 0, f"ELS needs even H,W; got {H}x{W}"
    blocks = img.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
    M = B * (H // 2) * (W // 2)
    flat = blocks.reshape(M, C, 2, 2).permute(0, 2, 3, 1).reshape(M, 4, C)
    diff = flat.unsqueeze(2) - flat.unsqueeze(1)
    dists = (diff ** 2).sum(dim=-1)
    eye = torch.eye(4, dtype=torch.bool, device=img.device).unsqueeze(0)
    dists = dists.masked_fill(eye, float("inf"))
    idx = torch.argmin(dists.reshape(M, -1), dim=1)
    p = idx // 4
    q = idx % 4
    swapped = flat.clone()
    ar = torch.arange(M, device=img.device)
    tmp = swapped[ar, p, :].clone()
    swapped[ar, p, :] = swapped[ar, q, :]
    swapped[ar, q, :] = tmp
    out = swapped.view(M, 2, 2, C).permute(0, 3, 1, 2)
    out = out.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return out.contiguous().view(B, C, H, W)


class ZSN2NNet(nn.Module):
    def __init__(self, n_chan: int, chan_embed: int = 48):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


def zsn2n_loss(model: nn.Module, noisy_img: torch.Tensor) -> torch.Tensor:
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))
    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
    return loss_res + loss_cons


def train_zsn2n(noisy: torch.Tensor, epochs: int, lr: float, step_size: int, gamma: float, seed: int) -> Tuple[torch.Tensor, float, int]:
    set_seed(seed)
    model = ZSN2NNet(noisy.shape[1]).to(noisy.device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        loss = zsn2n_loss(model, noisy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        den = (noisy - model(noisy)).clamp(0, 1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return den, elapsed, n_params


class RangeGatedKPNStage(nn.Module):
    def __init__(self, n_chan: int, chan: int = 16, k: int = 5, smooth_mix: float = 0.75, range_sigma_init: float = 0.08):
        super().__init__()
        self.k = k
        self.pad = k // 2
        self.smooth_mix = float(smooth_mix)
        self.net = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, 3, padding=1, groups=n_chan, bias=False),
            nn.Conv2d(n_chan, chan, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan, chan, 3, padding=1, groups=chan, bias=False),
            nn.Conv2d(chan, chan, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(chan, k * k, 1, bias=True),
        )
        with torch.no_grad():
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            final.bias[(k * k) // 2] = 3.0
        # trainable positive sigma via softplus parameterization
        init = max(float(range_sigma_init), 1e-4)
        inv_softplus = math.log(math.exp(init) - 1.0) if init < 20 else init
        self.log_sigma_r = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        spatial_w = F.softmax(self.net(x), dim=1)  # [B,K2,H,W]
        patches = F.unfold(x, kernel_size=self.k, padding=self.pad).reshape(B, C, self.k * self.k, H, W)
        center = x.unsqueeze(2)  # [B,C,1,H,W]
        # RGB-aware range distance: average squared distance across channels.
        dist2 = ((patches - center) ** 2).mean(dim=1)  # [B,K2,H,W]
        sigma = F.softplus(self.log_sigma_r).to(dtype=x.dtype, device=x.device) + 1e-6
        range_gate = torch.exp(-dist2 / (2.0 * sigma * sigma))
        w = spatial_w * range_gate
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        out = (patches * w.unsqueeze(1)).sum(dim=2)
        return (self.smooth_mix * out + (1.0 - self.smooth_mix) * x).contiguous()


class RangeGatedAdaptKPN(nn.Module):
    def __init__(self, n_chan: int = 3, chan: int = 16, k: int = 5, stages: int = 3, smooth_mix: float = 0.75, range_sigma_init: float = 0.08):
        super().__init__()
        self.stages = nn.ModuleList([
            RangeGatedKPNStage(n_chan=n_chan, chan=chan, k=k, smooth_mix=smooth_mix, range_sigma_init=range_sigma_init)
            for _ in range(stages)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for stage in self.stages:
            out = stage(out)
        return out.contiguous()


class F2NStyleLoss:
    def __init__(self, device: torch.device, lambda_edge: float = 350.0):
        self.kernel_size = 7
        self.sigma_narrow = 9.0
        self.sigma_wide = 10.0
        self.device = device
        self.lambda_edge = float(lambda_edge)
        coords = torch.arange(self.kernel_size, dtype=torch.float32, device=device) - self.kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g1 = torch.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma_narrow ** 2)); g1 = g1 / g1.sum()
        g2 = torch.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma_wide ** 2)); g2 = g2 / g2.sum()
        self.base_dog = (g1 - g2).view(1, 1, self.kernel_size, self.kernel_size)

    def __call__(self, noisy_input: torch.Tensor, model: nn.Module) -> torch.Tensor:
        C = noisy_input.shape[1]
        dog = self.base_dog.to(noisy_input.device, noisy_input.dtype).repeat(C, 1, 1, 1)
        n1, n2 = pair_downsampler(noisy_input)
        s1 = els(n1)
        s2 = els(n2)
        d1 = model(s1)
        d2 = model(s2)
        loss_resolution = (1.0 / 3.0) * F.l1_loss(d1, d2)
        d_full = model(noisy_input)
        dd1, dd2 = pair_downsampler(d_full)
        loss_cross = (1.0 / 3.0) * (F.l1_loss(d1, dd1) + F.l1_loss(d2, dd2))
        loss_denoise = (1.0 / 3.0) * F.l1_loss(dd1, dd2)
        e_noisy = F.conv2d(noisy_input, dog, padding=self.kernel_size // 2, groups=C)
        e_den = F.conv2d(d_full, dog, padding=self.kernel_size // 2, groups=C)
        loss_edge = self.lambda_edge * F.l1_loss(torch.abs(e_noisy), torch.abs(e_den))
        return loss_resolution + loss_cross + loss_denoise + loss_edge


def train_ours(noisy: torch.Tensor, args, seed: int) -> Tuple[torch.Tensor, float, int]:
    set_seed(seed)
    model = RangeGatedAdaptKPN(
        n_chan=noisy.shape[1],
        chan=args.kpn_chan,
        k=args.kpn_k,
        stages=args.kpn_stages,
        smooth_mix=args.smooth_mix,
        range_sigma_init=args.range_sigma_init,
    ).to(noisy.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr_ours, weight_decay=args.weight_decay)
    sch = OneCycleLR(opt, max_lr=args.lr_ours, epochs=args.epochs_ours, steps_per_epoch=1)
    loss_fn = F2NStyleLoss(noisy.device, lambda_edge=args.lambda_edge)
    t0 = time.time()
    model.train()
    for _ in range(args.epochs_ours):
        loss = loss_fn(noisy, model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        den = model(noisy).clamp(0, 1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return den, elapsed, n_params
