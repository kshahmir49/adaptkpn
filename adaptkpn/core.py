#!/usr/bin/env python3
"""Final frozen AdaptKPN core used by the CT and FMD experiments."""

import math
import random
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(device: torch.device, amp: bool):
    enabled = bool(device.type == "cuda" and amp)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    class _DummyScaler:
        def scale(self, loss):
            return loss
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
    return _DummyScaler()


def autocast_context(device: torch.device, amp: bool):
    enabled = bool(device.type == "cuda" and amp)
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        try:
            return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)
        except TypeError:
            pass
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


class _KPNStage(nn.Module):
    def __init__(
        self,
        n_chan: int,
        chan: int,
        k: int,
        smooth_mix: float,
        use_range_gate: bool = True,
        range_sigma_init: float = 0.08,
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

        init = max(float(range_sigma_init), 1e-6)
        self.log_sigma_r = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))

        with torch.no_grad():
            final_conv = self.net[-1]
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)
            final_conv.bias[(self.k * self.k) // 2] = 3.0

    def current_sigma_r(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_r).clamp(self.range_sigma_min, self.range_sigma_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        spatial_weights = F.softmax(self.net(x), dim=1)
        patches = F.unfold(x, kernel_size=self.k, padding=self.pad)
        patches = patches.reshape(b, c, self.k * self.k, h, w)

        if self.use_range_gate:
            center = x.unsqueeze(2)
            sigma = self.current_sigma_r().to(dtype=x.dtype, device=x.device)
            diff2 = (patches - center).pow(2).mean(dim=1)
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
        stages: int = 2,
        smooth_mix: float = 0.75,
        use_range_gate: bool = True,
        range_sigma_init: float = 0.08,
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
        return [float(s.current_sigma_r().detach().cpu().item()) for s in self.stages]


def pair_downsampler(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _, c, h, w = img.shape
    assert h % 2 == 0 and w % 2 == 0, f"Expected even H/W, got {h}x{w}"
    f1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    f2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    return (
        F.conv2d(img, f1, stride=2, groups=c).contiguous(),
        F.conv2d(img, f2, stride=2, groups=c).contiguous(),
    )


def ELS_local_shuffle(img: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self,
        device: torch.device,
        lambda_edge: float = 350.0,
        kernel_size: int = 7,
        sigma_narrow: float = 9.0,
        sigma_wide: float = 10.0,
    ):
        self.device = device
        self.lambda_edge = float(lambda_edge)
        self.kernel_size = int(kernel_size)
        coords = torch.arange(self.kernel_size, dtype=torch.float32, device=device) - self.kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g1 = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_narrow ** 2))
        g1 = g1 / g1.sum()
        g2 = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_wide ** 2))
        g2 = g2 / g2.sum()
        self.base_dog = (g1 - g2).unsqueeze(0).unsqueeze(0)

    def __call__(self, noisy: torch.Tensor, model: nn.Module) -> torch.Tensor:
        c = noisy.shape[1]
        dog = self.base_dog.to(device=noisy.device, dtype=noisy.dtype).repeat(c, 1, 1, 1)
        y1, y2 = pair_downsampler(noisy)
        p1 = model(ELS_local_shuffle(y1)).contiguous()
        p2 = model(ELS_local_shuffle(y2)).contiguous()
        loss_resolution = (1.0 / 3.0) * F.l1_loss(p1, p2)
        den_full = model(noisy).contiguous()
        d1, d2 = pair_downsampler(den_full)
        loss_cross_scale = (1.0 / 3.0) * (F.l1_loss(p1, d1) + F.l1_loss(p2, d2))
        loss_denoise = (1.0 / 3.0) * F.l1_loss(d1, d2)
        edges_noisy = F.conv2d(noisy, dog, padding=self.kernel_size // 2, groups=c)
        edges_den = F.conv2d(den_full, dog, padding=self.kernel_size // 2, groups=c)
        loss_edge = self.lambda_edge * F.l1_loss(torch.abs(edges_noisy), torch.abs(edges_den))
        return loss_resolution + loss_cross_scale + loss_denoise + loss_edge


def frozen_args(**overrides):
    cfg = dict(
        seed=123,
        amp=True,
        compile=False,
        epochs=500,
        lr=1.6e-2,
        optimizer="adamw",
        weight_decay=0.01,
        scheduler="onecycle",
        step_size=500,
        gamma=0.5,
        adapt_lambda_edge=350.0,
        kpn_chan=16,
        kpn_k=5,
        kpn_stages=2,
        smooth_mix=0.75,
        use_range_gate=True,
        range_sigma_init=0.08,
        range_sigma_min=0.005,
        range_sigma_max=0.25,
        use_ema=False,
        ema_decay=0.999,
        print_every=100,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def train_adaptkpn(noisy: torch.Tensor, args, device: torch.device):
    set_seed(args.seed)
    model = AdaptKPN(
        n_chan=noisy.shape[1],
        chan=args.kpn_chan,
        k=args.kpn_k,
        stages=args.kpn_stages,
        smooth_mix=args.smooth_mix,
        use_range_gate=args.use_range_gate,
        range_sigma_init=args.range_sigma_init,
        range_sigma_min=args.range_sigma_min,
        range_sigma_max=args.range_sigma_max,
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

    loss_fn = F2NStyleLoss(device, lambda_edge=args.adapt_lambda_edge)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.use_ema else None
    scaler = make_grad_scaler(device, args.amp)

    last_loss = 0.0
    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        with autocast_context(device, args.amp):
            loss = loss_fn(noisy, model)
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
            print(f"AdaptKPN epoch {epoch:4d}/{args.epochs} loss={last_loss:.6f} lr={opt.param_groups[0]['lr']:.2e}")

    if ema is not None:
        model.load_state_dict({k: v.to(device) for k, v in ema.items()})
    model.eval()
    return model, last_loss
