"""
MFU profiler for pre-tokenized training data.

Measures end-to-end effective throughput (memmap batch read +
GPU forward+backward) and reports Model Flop Utilization (MFU).

torch.compile() warmup: the first forward pass triggers JIT compilation
and is not representative. Use --n-warmup (default 20) to skip enough
iterations before timing begins. With compile the warmup phase will take
noticeably longer — this is expected, not a hang.

Usage:
    python -m src.utils.profile_throughput \\
        --config configs/training.yaml \\
        --n-warmup 20 \\
        --n-batches 100
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.dataset import TokenizedDataset
from src.train import load_training_config, resolve_device
from src.transformer import (
    LanguageModel,
    ModelConfig,
    compile_language_model,
    forward_backward_micro_step,
    get_supported_weights_precision,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants and helpers
# ──────────────────────────────────────────────────────────────────────────────

GPU_TFLOPS: dict[str, float] = {
    "A100_40GB": 312.0,
    "A100_80GB": 312.0,
    "H100_SXM":  989.0,
    "H100_PCIe": 756.0,
    "RTX_4090":  330.0,
    "RTX_3090":  142.0,
    "V100_FP16":  14.0,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _fmt(value: float) -> str:
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:.2f}"


def _next_batch(dataset: TokenizedDataset, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the next batch, resetting the split cursor if exhausted."""
    try:
        return dataset.get_sequential_batch(split)
    except StopIteration:
        dataset.reset_split(split)
        return dataset.get_sequential_batch(split)


# ──────────────────────────────────────────────────────────────────────────────
# Core measurement
# ──────────────────────────────────────────────────────────────────────────────

def _run(
    model: torch.nn.Module,
    scaler: torch.amp.GradScaler,
    dataset: TokenizedDataset,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    n_warmup: int,
    n_batches: int,
    compiled: bool,
) -> float:
    """
    Return mean wall-clock seconds per training step (memmap read + fwd + bwd).

    Warmup iterations are run first and excluded from timing. When
    torch.compile is active, the first several warmup steps trigger JIT
    compilation; subsequent steps settle to the compiled speed.
    """
    label = "warmup (compile + CUDA)" if compiled else "warmup"
    print(f"  Running {n_warmup} {label} iterations …", flush=True)

    for _ in range(n_warmup):
        x, y = _next_batch(dataset, "train")
        x, y = x.to(device), y.to(device)
        forward_backward_micro_step(model, scaler, x, y, device, amp_dtype, use_amp)
    _sync(device)

    print(f"  Timing {n_batches} iterations …", flush=True)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        x, y = _next_batch(dataset, "train")
        x, y = x.to(device), y.to(device)
        forward_backward_micro_step(model, scaler, x, y, device, amp_dtype, use_amp)
    _sync(device)

    return (time.perf_counter() - t0) / n_batches


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _report(
    *,
    model: torch.nn.Module,
    model_config: ModelConfig,
    gpu: str | None,
    tokens_per_batch: int,
    step_s: float,
    compiled: bool,
) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6 * n_params

    throughput = tokens_per_batch / step_s

    theoretical_flops_s = GPU_TFLOPS.get(gpu, 312.0) * 1e12 if gpu else None
    mfu = (
        throughput * flops_per_token / theoretical_flops_s * 100
        if theoretical_flops_s is not None
        else None
    )

    target_tokens = 6.4e9
    hours = target_tokens / throughput / 3600

    print()
    print("=" * 56)
    print("  MFU PROFILE")
    print("=" * 56)
    print(f"  Model parameters  : {_fmt(n_params)} ({n_params:,})")
    print(f"  Tokens per batch  : {_fmt(tokens_per_batch)}")
    print(f"  torch.compile     : {'yes' if compiled else 'no'}")
    print(f"  GPU               : {gpu or 'unknown'}")
    print()
    print(f"  Step time         : {step_s * 1e3:,.1f} ms")
    print(f"  Throughput        : {_fmt(throughput)} tok/s")
    if mfu is not None:
        print(f"  MFU               : {mfu:.1f}%")
    print()
    print(f"  Time to train 6.4 B tokens : {hours:,.0f} h  ({hours / 24:.1f} days)")
    print("=" * 56)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def profile_throughput(
    config_path: Path,
    n_warmup: int,
    n_batches: int,
    gpu: str | None,
) -> None:
    config       = load_training_config(config_path)
    model_config = ModelConfig.from_yaml(config.model_config)
    device       = resolve_device(config.device)
    amp_dtype    = get_supported_weights_precision(device)
    use_amp      = device.type == "cuda"
    use_scaler   = use_amp and amp_dtype == torch.float16

    print(f"\nLoading tokenized dataset from {config.data_dir} …")
    dataset = TokenizedDataset(
        data_dir=config.data_dir,
        sequence_length=model_config.sequence_length,
        batch_size=config.batch_size,
    )

    print(f"Building model on {device} …")
    model: torch.nn.Module = LanguageModel.from_config(
        model_config, kv_cache={}, device=device,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    model = compile_language_model(model, enabled=config.compile_model)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    print("Profiling …")
    step_s = _run(
        model=model,
        scaler=scaler,
        dataset=dataset,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        n_warmup=n_warmup,
        n_batches=n_batches,
        compiled=config.compile_model,
    )

    tokens_per_batch = config.batch_size * model_config.sequence_length
    _report(
        model=model,
        model_config=model_config,
        gpu=gpu,
        tokens_per_batch=tokens_per_batch,
        step_s=step_s,
        compiled=config.compile_model,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile effective MFU on pre-tokenized training data."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument(
        "--n-warmup", type=int, default=20,
        help=(
            "Iterations to run before timing starts. "
            "Covers torch.compile JIT and CUDA kernel warm-up. "
            "Increase to 50+ if compile is enabled and the first report looks too slow."
        ),
    )
    parser.add_argument(
        "--n-batches", type=int, default=100,
        help="Number of timed training steps.",
    )
    parser.add_argument(
        "--gpu", type=str, default="A100_40GB",
        help=f"GPU model for MFU calculation. Known: {', '.join(GPU_TFLOPS)}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    profile_throughput(args.config, args.n_warmup, args.n_batches, args.gpu)
