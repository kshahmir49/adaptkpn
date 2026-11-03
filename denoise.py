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
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

try:
    from skimage.metrics import structural_similarity
except Exception:
    structural_similarity = None


class KPNStage(nn.Module):
    """One adaptive kernel-prediction stage with a per-pixel blend.

    The head predicts k*k kernel logits and one blend logit per pixel.
    output = alpha * (softmax-weighted k x k average) + (1 - alpha) * input.
    If fixed_blend is set, alpha is the constant fixed_blend and the head
    predicts kernel logits only (the original fixed 0.8/0.2 variant).
    """

    def __init__(self, channels=1, hidden=16, kernel_size=5, blend_init=-1.0, fixed_blend=None):
        super().__init__()
        self.k = kernel_size
        self.kk = kernel_size * kernel_size
        self.pad = kernel_size // 2
        self.fixed_blend = fixed_blend
        out_ch = self.kk if fixed_blend is not None else self.kk + 1

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1),
        )
        with torch.no_grad():
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            last.bias[self.kk // 2] = 3.0  # warm start near identity
            if fixed_blend is None:
                last.bias[self.kk] = blend_init

    def forward(self, x):
        b, c, h, w = x.shape
        raw = self.net(x)
        weights = F.softmax(raw[:, :self.kk], dim=1)
        patches = F.unfold(x, self.k, padding=self.pad).view(b, c, self.kk, h, w)
        avg = (patches * weights.unsqueeze(1)).sum(dim=2)
        alpha = self.fixed_blend if self.fixed_blend is not None else torch.sigmoid(raw[:, self.kk:])
        return (alpha * avg + (1.0 - alpha) * x).contiguous()


class AdaptKPN(nn.Module):
    """Three-stage cascade of KPNStage (about 2.7k parameters at the defaults)."""

    def __init__(self, channels=1, hidden=16, kernel_size=5, stages=3, blend_init=-1.0, fixed_blend=None):
        super().__init__()
        self.stages = nn.ModuleList([
            KPNStage(channels, hidden, kernel_size, blend_init, fixed_blend)
            for _ in range(stages)
        ])

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x.contiguous()


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


def zs_n2n_loss(model, y):
    """Zero-shot Noise2Noise loss (pair downsampling): residual + consistency."""
    y1, y2 = pair_downsample(y)
    p1, p2 = model(y1), model(y2)
    d1, d2 = pair_downsample(model(y))
    loss_res = 0.5 * (F.mse_loss(y1, p2) + F.mse_loss(y2, p1))
    loss_con = 0.5 * (F.mse_loss(p1, d1) + F.mse_loss(p2, d2))
    return loss_res + loss_con


def poisson_thinning_loss(model, y, peak, p=0.5):
    """Auxiliary Poisson-thinning Noise2Noise loss on pseudo photon counts."""
    counts = (y.clamp(0, 1) * peak).contiguous()
    c1 = torch.binomial(counts, torch.full_like(counts, p))
    c2 = counts - c1
    x1 = c1 / (p * peak)
    x2 = c2 / ((1.0 - p) * peak)
    return 0.5 * (F.mse_loss(model(x1), x2) + F.mse_loss(model(x2), x1))


def _transform(x, t):
    if t < 4:
        return torch.rot90(x, k=t, dims=(-2, -1)).contiguous()
    return torch.flip(torch.rot90(x, k=t - 4, dims=(-2, -1)), dims=(-1,)).contiguous()


def _inverse_transform(x, t):
    if t < 4:
        return torch.rot90(x, k=(4 - t) % 4, dims=(-2, -1)).contiguous()
    return torch.rot90(torch.flip(x, dims=(-1,)), k=(4 - (t - 4)) % 4, dims=(-2, -1)).contiguous()


@torch.no_grad()
def denoise_self_ensemble(model, y, enabled=True):
    """Average the model output over the 8 rotations/flips of the input."""
    if not enabled:
        return model(y).contiguous()
    acc = torch.zeros_like(y)
    for t in range(8):
        acc = acc + _inverse_transform(model(_transform(y, t)), t)
    return (acc / 8.0).contiguous()


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

    model = AdaptKPN(
        channels=noisy.shape[1],
        hidden=args.hidden,
        kernel_size=args.kernel_size,
        stages=args.stages,
        blend_init=args.blend_init,
        fixed_blend=args.fixed_blend,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema else None

    last = {}
    model.train()
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = zs_n2n_loss(model, noisy)
        if args.poisson_weight > 0:
            loss = loss + args.poisson_weight * poisson_thinning_loss(model, noisy, args.poisson_peak, args.poisson_p)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema_decay).add_(v.detach(), alpha=1.0 - args.ema_decay)

        last = {"loss": float(loss.detach().cpu())}
        if args.print_every and (epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs):
            print(f"epoch {epoch:4d}/{args.epochs} loss={last['loss']:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")

    if ema is not None:
        model.load_state_dict(ema)
    model.eval()
    den = denoise_self_ensemble(model, noisy, enabled=args.self_ensemble).clamp(0, 1).contiguous()

    last["seconds"] = time.time() - start
    last["params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
        result["adaptkpn_psnr"] = psnr(den, target)
        result["adaptkpn_ssim"] = ssim(den, target)
        print(f"{tag}: noisy {result['noisy_psnr']:.2f}/{result['noisy_ssim']:.4f}  AdaptKPN {result['adaptkpn_psnr']:.2f}/{result['adaptkpn_ssim']:.4f}")
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
    p = argparse.ArgumentParser(description="AdaptKPN zero-shot denoising.")
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

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--step-size", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--blend-init", type=float, default=-1.0,
                   help="Initial blend logit (sigmoid). Natural images used -3.0/-1.5/-0.5 for sigma 10/25/50; D45 used -1.0.")
    p.add_argument("--fixed-blend", type=float, default=None,
                   help="Use a constant blend (e.g. 0.8) instead of the learned per-pixel blend. 0.8 gives the original B30 headline setup.")
    p.add_argument("--poisson-weight", type=float, default=0.25, help="0 disables the Poisson thinning loss.")
    p.add_argument("--poisson-peak", type=float, default=50.0)
    p.add_argument("--poisson-p", type=float, default=0.5)
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--self-ensemble", action="store_true", default=True)
    p.add_argument("--no-self-ensemble", dest="self_ensemble", action="store_false")
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
