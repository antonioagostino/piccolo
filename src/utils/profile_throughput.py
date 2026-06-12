from typing import cast
import argparse
import time
from pathlib import Path

import torch

from src.dataset import TokenizedPreTrainingDataset
from src.train import load_training_config, validate_config, validate_device
from src.transformer import (
    LanguageModel,
    get_supported_weights_precision,
)

GPU_TFLOPS: dict[str, float] = {
    "A100_40GB": 312.0,
    "A100_80GB": 312.0,
    "H100_SXM":  989.0,
    "H100_PCIe": 756.0,
    "RTX_4090":  330.0,
    "RTX_3090":  142.0,
    "V100_FP16":  14.0,
}

def format_output(value: float) -> str:
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:.2f}"

def profile_throughput(
    config_path: Path,
    n_warmup: int,
    n_batches: int,
    gpu: str | None,
    target_tokens: float,
) -> None:
    config = load_training_config(config_path)
    validate_config(config)
    device = validate_device(config["device"])
    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16
    torch.set_float32_matmul_precision("high")

    language_model: LanguageModel = LanguageModel.from_config(
        config["model_config"],
        kv_cache={},
        device=device,
        gradient_checkpointing=config["gradient_checkpointing"],
    ).to(device)

    if config["compile_model"]:
        language_model = cast(LanguageModel, torch.compile(language_model))
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    dataset = TokenizedPreTrainingDataset(
        data_dir=config["data_dir"],
        sequence_length=language_model.sequence_length,
        batch_size=config["batch_size"],
    )

    print("Start profiling...")

    for _ in range(n_warmup):
        language_model.zero_grad(set_to_none=True)
        try:
            x, y = dataset.get_sequential_batch("train")
        except StopIteration:
            dataset.reset_split("train")
            x, y = dataset.get_sequential_batch("train")

        x, y = x.to(device), y.to(device)
        language_model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = language_model(inputs, targets=targets)

        loss_scale = 1 / config["gradient_accumulation_steps"]
        scaler.scale(loss * loss_scale).backward()
        scaler.update()
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_batches):
        language_model.zero_grad(set_to_none=True)
        try:
            x, y = dataset.get_sequential_batch("train")
        except StopIteration:
            dataset.reset_split("train")
            x, y = dataset.get_sequential_batch("train")

        x, y = x.to(device), y.to(device)
        language_model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = language_model(inputs, targets=targets)

        loss_scale = 1 / config["gradient_accumulation_steps"]
        scaler.scale(loss * loss_scale).backward()
        scaler.update()
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    step_s = (time.perf_counter() - t0) / n_batches

    tokens_per_batch = config["batch_size"] * language_model.sequence_length
    n_params = sum(p.numel() for p in language_model.parameters())
    flops_per_token = 6 * n_params

    throughput = tokens_per_batch / step_s

    theoretical_flops_s = GPU_TFLOPS.get(gpu, 312.0) * 1e12 if gpu else None
    mfu = (
        throughput * flops_per_token / theoretical_flops_s * 100
        if theoretical_flops_s is not None
        else None
    )

    hours = target_tokens / throughput / 3600

    print(f"Model parameters: {format_output(n_params)} ({n_params:,})")
    print(f"Tokens per batch: {format_output(tokens_per_batch)}")
    print(f"torch.compile: {'yes' if config["compile_model"] else 'no'}")
    print(f"GPU: {gpu or 'unknown'}")
    print(f"Step time: {step_s * 1e3:,.1f} ms")
    print(f"Throughput: {format_output(throughput)} tok/s")
    if mfu is not None:
        print(f"MFU: {mfu:.1f}%")
    print()
    print(f"Time to train {format_output(target_tokens)} tokens : {hours:,.0f} h  ({hours / 24:.1f} days)")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute effective MFU using pre-training tokenized data."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument(
        "--n-warmup", type=int, default=20,
        help=(
            "Iterations to run before timing starts. "
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
    parser.add_argument(
        "--target-tokens", type=float, default=6.4e9,
        help="Number of tokens to project training time for (default: 6.4B).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_throughput(args.config, args.n_warmup, args.n_batches, args.gpu, args.target_tokens)
