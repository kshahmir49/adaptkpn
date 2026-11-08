#!/usr/bin/env python3
import argparse
import csv
import gc
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch

from adaptkpn import frozen_args, train_adaptkpn


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_fixed4095(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    arr = np.clip(arr / 4095.0, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_adapt(x_cpu, device, seed):
    x = x_cpu.to(device)
    cfg = frozen_args(seed=seed, epochs=500, print_every=1000000)
    sync(device)
    t0 = time.perf_counter()
    model, _ = train_adaptkpn(x, cfg, device)
    with torch.no_grad():
        den = model(x).clamp(0, 1)
    sync(device)
    elapsed = time.perf_counter() - t0
    del model, den, x
    cleanup()
    return elapsed


def run_f2n(f2n, x_cpu, device, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x = x_cpu.to(device)
    sync(device)
    t0 = time.perf_counter()
    model = f2n.DenoisingPipeline(num_stages=2, patch_size=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=500, steps_per_epoch=1)
    loss_function = f2n.LossFunction(device, lambda_=400)
    f2n.train_model(model, x, loss_function, optimizer, scheduler, epochs=500)
    model.eval()
    with torch.no_grad():
        den = model(x)
    sync(device)
    elapsed = time.perf_counter() - t0
    del model, optimizer, scheduler, loss_function, den, x
    cleanup()
    return elapsed


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Same-GPU CT timing benchmark against the official F2N implementation")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--f2n-path", required=True, help="Path to official Filter2Noise/filter2noise.py")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="ct_runtime_results.csv")
    ap.add_argument("--paper-f2n-seconds", type=float, default=16.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for timing benchmark")
    device = torch.device("cuda")
    f2n = import_module(args.f2n_path, "official_f2n")
    df = pd.read_csv(args.manifest).sort_values("task_id").reset_index(drop=True)
    need = args.warmup + args.n
    if len(df) < need:
        raise ValueError(f"Manifest has {len(df)} rows but benchmark needs {need}")

    print("GPU", torch.cuda.get_device_name(0))
    print("Timer scope model creation + per-image optimization + final forward")
    print("Input loading and metric computation excluded")

    for i in range(args.warmup):
        x = load_fixed4095(df.iloc[i].low_path)
        run_adapt(x, device, args.seed + i)
        run_f2n(f2n, x, device, args.seed + 10000 + i)

    rows = []
    measured = df.iloc[args.warmup:need].reset_index(drop=True)
    for j, row in measured.iterrows():
        x = load_fixed4095(row.low_path)
        order = ["adaptkpn", "f2n"] if j % 2 == 0 else ["f2n", "adaptkpn"]
        times = {}
        for method in order:
            if method == "adaptkpn":
                times[method] = run_adapt(x, device, args.seed + j)
            else:
                times[method] = run_f2n(f2n, x, device, args.seed + 10000 + j)
        for method in ["adaptkpn", "f2n"]:
            rows.append({
                "image_index": j,
                "task_id": row.get("task_id", j),
                "method": method,
                "seconds": times[method],
                "gpu": torch.cuda.get_device_name(0),
                "normalization": "fixed4095",
                "timer_scope": "model_init+optimization+final_forward",
            })
        write_csv(args.out, rows)

    out = pd.DataFrame(rows)
    summary = out.groupby("method").seconds.agg(["count", "mean", "std", "median", "min", "max"])
    print("
Timing summary")
    print(summary.to_string())
    f2n_mean = float(summary.loc["f2n", "mean"])
    adapt_mean = float(summary.loc["adaptkpn", "mean"])
    scale = args.paper_f2n_seconds / f2n_mean
    print("
Hardware-normalized AdaptKPN estimate", adapt_mean * scale)
    print("Measured AdaptKPN/F2N ratio", adapt_mean / f2n_mean)


if __name__ == "__main__":
    main()
