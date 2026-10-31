"""Check the GPU setup and time one training step of the model, without data.

Run this when training is slower than expected. It prints the torch/CUDA/GPU
versions and the seconds per step for fp16 vs fp32 and channels_last on/off,
so you can see whether the GPU itself is slow or the setup is wrong.

    python scripts/benchmark_gpu.py --config configs/baseline_aptos.yaml

On a Kaggle T4, EfficientNet-B3 at 512 px with batch 16 and fp16 should take
roughly 0.3-0.7 s per training step on one GPU.
"""
import argparse
import time

import _bootstrap  # noqa: F401
import torch

from retina.config import load_config
from retina.engine import environment_info
from retina.model import build_model


def time_step(model, x, y, amp: bool, channels_last: bool, steps: int, train: bool) -> float:
    device = x.device
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    model.to(memory_format=fmt)
    x = x.contiguous(memory_format=fmt)
    opt = torch.optim.SGD(model.parameters(), lr=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    def one():
        if train:
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = (model(x).float() - y).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
                model(x)

    model.train(train)
    print(f"  running {'train' if train else 'infer'} fp16={amp} channels_last={channels_last} ...", flush=True)
    for _ in range(3):  # warm-up (cuDNN autotuning, memory allocation)
        one()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        one()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--try-data-parallel", action="store_true",
                    help="Also time nn.DataParallel over all GPUs (segfaults on Kaggle T4 x2 with torch 2.11)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    print("Environment:", environment_info())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    size, bs = cfg["train"]["image_size"], cfg["train"]["batch_size"]
    model = build_model(cfg["model"], pretrained=False).to(device)
    x = torch.randn(bs, 3, size, size, device=device)
    y = torch.rand(bs, device=device) * 4
    print(f"Model {cfg['model']['backbone']}, batch {bs}, {size}x{size}, device {device} (single GPU)\n")

    combos = [(True, True), (True, False), (False, False)] if device.type == "cuda" else [(False, False)]
    print(f"{'mode':<10}{'fp16':<7}{'chan_last':<11}{'s/step':>8}{'img/s':>9}")
    for train in (True, False):
        for amp, cl in combos:
            try:
                s = time_step(model, x, y, amp, cl, args.steps, train)
                print(f"{'train' if train else 'infer':<10}{str(amp):<7}{str(cl):<11}{s:>8.3f}{bs / s:>9.1f}")
            except RuntimeError as e:  # e.g. out of memory
                print(f"{'train' if train else 'infer':<10}{str(amp):<7}{str(cl):<11}  failed: {str(e).splitlines()[0]}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if args.try_data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print("\nDataParallel over", torch.cuda.device_count(), "GPUs (fp16, channels_last):")
        dp = torch.nn.DataParallel(model)
        for train in (True, False):
            try:
                s = time_step(dp, x, y, True, True, args.steps, train)
                print(f"{'train' if train else 'infer':<10}{'True':<7}{'True':<11}{s:>8.3f}{bs / s:>9.1f}")
            except RuntimeError as e:
                print(f"{'train' if train else 'infer'}  failed: {str(e).splitlines()[0]}")
    if device.type == "cuda":
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
