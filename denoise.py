#!/usr/bin/env python3

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

try:
    from skimage.metrics import structural_similarity
except Exception:
    structural_similarity = None


class RangeGatedStage(nn.Module):
    def __init__(self, channels=1, hidden=16, kernel_size=5, mix=0.75, sigma=0.08):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.mix = mix

        self.predictor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, kernel_size * kernel_size, 1),
        )

        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma), dtype=torch.float32))
        self._init_identity()

    def _init_identity(self):
        layer = self.predictor[-1]
        with torch.no_grad():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer.bias[(self.kernel_size * self.kernel_size) // 2] = 3.0

    def range_sigma(self):
        return torch.exp(self.log_sigma).clamp(0.005, 0.25)

    def forward(self, x):
        b, c, h, w = x.shape
        k2 = self.kernel_size * self.kernel_size

        spatial = F.softmax(self.predictor(x), dim=1)
        patches = F.unfold(x, self.kernel_size, padding=self.pad).view(b, c, k2, h, w)

        dist2 = (patches - x.unsqueeze(2)).pow(2).mean(dim=1)
        sigma = self.range_sigma().to(device=x.device, dtype=x.dtype)
        range_gate = torch.exp(-dist2 / (2.0 * sigma * sigma + 1e-12))

        kernel = spatial * range_gate
        kernel = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-12)

        filtered = (patches * kernel.unsqueeze(1)).sum(dim=2)
        return ((1.0 - self.mix) * x + self.mix * filtered).contiguous()


class RangeGatedDenoiser(nn.Module):
    def __init__(self, channels=1, hidden=16, kernel_size=5, stages=3, mix=0.75, sigma=0.08):
        super().__init__()
        self.stages = nn.ModuleList([
            RangeGatedStage(channels, hidden, kernel_size, mix, sigma)
            for _ in range(stages)
        ])

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x.contiguous()

    def sigmas(self):
        return [float(stage.range_sigma().detach().cpu()) for stage in self.stages]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def even_crop(x):
    h, w = x.shape[-2:]
    return x[..., :h - h % 2, :w - w % 2].contiguous()


def pair_downsample(x):
    x = even_crop(x)
    c = x.shape[1]
    f1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    f2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    return (
        F.conv2d(x, f1, stride=2, groups=c).contiguous(),
        F.conv2d(x, f2, stride=2, groups=c).contiguous(),
    )


def local_shuffle(x):
    x = even_crop(x)
    b, c, h, w = x.shape

    blocks = x.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
    flat = blocks.reshape(-1, c, 2, 2).permute(0, 2, 3, 1).reshape(-1, 4, c)

    d = (flat.unsqueeze(2) - flat.unsqueeze(1)).pow(2).sum(dim=-1)
    eye = torch.eye(4, device=x.device, dtype=torch.bool).unsqueeze(0)
    d = d.masked_fill(eye, float("inf"))

    idx = d.reshape(flat.shape[0], -1).argmin(dim=1)
    p = idx // 4
    q = idx % 4
    r = torch.arange(flat.shape[0], device=x.device)

    out = flat.clone()
    tmp = out[r, p].clone()
    out[r, p] = out[r, q]
    out[r, q] = tmp

    out = out.view(-1, 2, 2, c).permute(0, 3, 1, 2)
    out = out.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return out.contiguous().view(b, c, h, w)


class ZeroShotLoss(nn.Module):
    def __init__(self, lambda_edge=350.0):
        super().__init__()
        self.lambda_edge = lambda_edge

        size = 7
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g1 = torch.exp(-(xx * xx + yy * yy) / (2 * 9.0 * 9.0))
        g2 = torch.exp(-(xx * xx + yy * yy) / (2 * 10.0 * 10.0))
        g1 = g1 / g1.sum()
        g2 = g2 / g2.sum()
        self.register_buffer("dog", (g1 - g2).view(1, 1, size, size))

    def edges(self, x):
        c = x.shape[1]
        kernel = self.dog.to(x.device, x.dtype).repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, padding=3, groups=c).abs()

    def forward(self, noisy, model):
        y1, y2 = pair_downsample(noisy)
        v1 = local_shuffle(y1)
        v2 = local_shuffle(y2)

        p1 = model(v1)
        p2 = model(v2)

        den = model(noisy)
        d1, d2 = pair_downsample(den)

        view = F.l1_loss(p1, p2)
        cross = 0.5 * (F.l1_loss(p1, d1) + F.l1_loss(p2, d2))
        denoise = F.l1_loss(d1, d2)
        edge = F.l1_loss(self.edges(den), self.edges(noisy))

        loss = (view + cross + denoise) / 3.0 + self.lambda_edge * edge
        return loss, {
            "loss": float(loss.detach().cpu()),
            "view": float(view.detach().cpu()),
            "cross": float(cross.detach().cpu()),
            "denoise": float(denoise.detach().cpu()),
            "edge": float(edge.detach().cpu()),
        }


def load_image(path, device, channels="gray", window_min=-3000.0, window_max=3000.0):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".dcm", ".ima"}:
        import pydicom
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        arr = (np.clip(hu, window_min, window_max) - window_min) / (window_max - window_min)
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device).contiguous()

    if suffix == ".npy":
        arr = np.load(path).astype(np.float32).squeeze()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D .npy image: {path}")
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device).contiguous()

    img = Image.open(path)
    if channels == "rgb":
        img = img.convert("RGB")
        arr = normalize_array(np.asarray(img))
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    else:
        img = img.convert("L")
        arr = normalize_array(np.asarray(img))
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return x.to(device=device, dtype=torch.float32).contiguous()


def normalize_array(arr):
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    else:
        out = arr.astype(np.float32)
        mx = float(np.nanmax(out)) if out.size else 1.0
        if mx > 1.5:
            out = out / mx
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def save_image(x, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().clamp(0, 1).cpu()[0]

    if x.shape[0] == 1:
        arr = (x[0].numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)
    else:
        arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr, mode="RGB").save(path)


def psnr(x, y):
    mse = torch.mean((x.detach().clamp(0, 1) - y.detach().clamp(0, 1)) ** 2).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def ssim(x, y):
    if structural_similarity is None:
        return float("nan")
    x = x.detach().clamp(0, 1).cpu()[0]
    y = y.detach().clamp(0, 1).cpu()[0]
    if x.shape[0] == 1:
        return float(structural_similarity(y[0].numpy(), x[0].numpy(), data_range=1.0))
    return float(structural_similarity(
        y.permute(1, 2, 0).numpy(),
        x.permute(1, 2, 0).numpy(),
        data_range=1.0,
        channel_axis=-1,
    ))


def train(noisy, args, device):
    seed_everything(args.seed)
    noisy = even_crop(noisy.to(device=device, dtype=torch.float32))

    model = RangeGatedDenoiser(
        channels=noisy.shape[1],
        hidden=args.hidden,
        kernel_size=args.kernel_size,
        stages=args.stages,
        mix=args.mix,
        sigma=args.sigma,
    ).to(device)

    criterion = ZeroShotLoss(lambda_edge=args.lambda_edge).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=1)

    last = {}
    model.train()
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss, parts = criterion(noisy, model)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        last = parts

        if args.print_every and (epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs):
            print(f"epoch {epoch:4d}/{args.epochs} loss={parts['loss']:.6f} edge={parts['edge']:.6f}")

    model.eval()
    with torch.no_grad():
        den = model(noisy).clamp(0, 1).contiguous()

    last["seconds"] = time.time() - start
    last["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    last["sigma_r"] = ";".join(f"{v:.6f}" for v in model.sigmas())
    return den, last


def run_pair(row, args, device, index=0):
    tag = row.get("tag") or row.get("image") or row.get("pair_key") or f"item_{index:05d}"
    dataset = row.get("dataset", args.dataset)
    channels = row.get("channels", "gray" if dataset.startswith("mayo") else args.channels)

    window_min = float(row.get("window_min", args.window_min))
    window_max = float(row.get("window_max", args.window_max))

    noisy_path = row.get("noisy") or row.get("low_path") or row.get("raw_path")
    target_path = row.get("target") or row.get("full_path") or row.get("gt_path")

    noisy = load_image(noisy_path, device, channels=channels, window_min=window_min, window_max=window_max)
    target = None if not target_path else even_crop(load_image(target_path, device, channels=channels, window_min=window_min, window_max=window_max))

    den, info = train(noisy, args, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_images:
        save_image(den, out_dir / f"{safe_name(tag)}_denoised.png")

    result = {
        "index": index,
        "dataset": dataset,
        "tag": tag,
        "noisy": noisy_path,
        "target": target_path or "",
        **info,
    }

    if target is not None:
        noisy_even = even_crop(noisy)
        result["noisy_psnr"] = psnr(noisy_even, target)
        result["noisy_ssim"] = ssim(noisy_even, target)
        result["ours_psnr"] = psnr(den, target)
        result["ours_ssim"] = ssim(den, target)
        print(f"{tag}: noisy {result['noisy_psnr']:.2f}/{result['noisy_ssim']:.4f}  ours {result['ours_psnr']:.2f}/{result['ours_ssim']:.4f}")
    else:
        print(f"{tag}: done")

    return result


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_results(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def build_single_row(args):
    if not args.noisy:
        raise ValueError("Use --noisy for one-image mode, or --manifest for dataset mode.")
    return {
        "dataset": args.dataset,
        "tag": args.tag or Path(args.noisy).stem,
        "noisy": args.noisy,
        "target": args.target or "",
        "channels": args.channels,
        "window_min": args.window_min,
        "window_max": args.window_max,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Range-gated zero-shot denoising.")
    p.add_argument("--dataset", choices=["mayo-b30", "mayo-d45", "fmd", "image"], default="image")
    p.add_argument("--manifest", default=None)
    p.add_argument("--noisy", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--tag", default=None)

    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--save-images", action="store_true", default=True)
    p.add_argument("--no-save-images", dest="save_images", action="store_false")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--channels", choices=["gray", "rgb"], default="gray")
    p.add_argument("--window-min", type=float, default=-3000.0)
    p.add_argument("--window-max", type=float, default=3000.0)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--mix", type=float, default=0.75)
    p.add_argument("--sigma", type=float, default=0.08)
    p.add_argument("--lambda-edge", type=float, default=350.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--print-every", type=int, default=100)

    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    rows = read_manifest(args.manifest) if args.manifest else [build_single_row(args)]
    rows = rows[args.start:]
    if args.limit is not None:
        rows = rows[:args.limit]

    results = []
    for i, row in enumerate(rows, start=args.start):
        try:
            results.append(run_pair(row, args, device, index=i))
            out_csv = args.out_csv or str(Path(args.out_dir) / "results.csv")
            write_results(out_csv, results)
        except Exception as e:
            error_row = {"index": i, "status": "error", "error": repr(e), **row}
            results.append(error_row)
            out_csv = args.out_csv or str(Path(args.out_dir) / "results.csv")
            write_results(out_csv, results)
            print(f"ERROR index={i}: {e}")

    print(f"saved results to {args.out_csv or Path(args.out_dir) / 'results.csv'}")


if __name__ == "__main__":
    main()
