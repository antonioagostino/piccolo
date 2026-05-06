from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.dataset import SplitTokenCounts, count_pre_training_tokens
from src.tokenizer import TiktokenTokenizer
from src.train import TrainingConfig, resolve_data_dir, validate_config
from src.transformer import ModelConfig

GPU_TFLOPS: dict[str, float] = {
    "A100_40GB": 312.0,
    "A100_80GB": 312.0,
}

# Advertised HBM capacity in GiB
GPU_MEMORY_GiB: dict[str, int] = {
    "A100_40GB": 40,
    "A100_80GB": 80,
}

MFU = 0.45
SECONDS_PER_DAY = 86_400
# bf16 model weights + bf16 gradients + fp32 Adam first moment + fp32 Adam second moment
BYTES_PER_PARAM = 2 + 2 + 4 + 4  # = 12


@dataclass(frozen=True)
class ChinchillaEstimate:
    gpu: str
    days: float
    theoretical_flops: float
    compute_budget: float
    n_opt: float
    d_opt: float


@dataclass(frozen=True)
class DatasetGapEstimate:
    train_tokens: int
    val_tokens: int
    n_files: int
    avg_tokens_per_file: float
    files_needed: int


@dataclass(frozen=True)
class ModelFitEstimate:
    n_params: int
    max_batch_size: int
    tokens_per_iteration: int
    train_iterations: int
    val_iterations: int


@dataclass(frozen=True)
class TokenBudgetEstimate:
    split_token_counts: SplitTokenCounts
    tokens_per_iteration: int
    train_iterations: int
    val_iterations: int
    chinchilla: Optional[ChinchillaEstimate] = None
    dataset_gap: Optional[DatasetGapEstimate] = None
    model_fit: Optional[ModelFitEstimate] = None


def _fmt(value: float) -> str:
    """
    Format a large number with K/M/B/T suffix.
    """
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return str(int(value))


def estimate_chinchilla(gpu: str, days: float) -> ChinchillaEstimate:
    """
    Compute Chinchilla-optimal model size and token count for a GPU rental.

    Args:
        gpu (str): GPU model identifier (e.g. ``"A100_40GB"``).
        days (float): Number of days the GPU will be rented.

    Returns:
        ChinchillaEstimate: Dataclass with compute budget, optimal parameter
            count (N_opt), and optimal token count (D_opt).

    Raises:
        ValueError: If the GPU model is not in the supported list.
    """
    if gpu not in GPU_TFLOPS:
        raise ValueError(f"Unknown GPU '{gpu}'. Valid options: {list(GPU_TFLOPS)}")
    theoretical_flops = GPU_TFLOPS[gpu] * 1e12
    compute_budget = theoretical_flops * MFU * days * SECONDS_PER_DAY
    n_opt = math.sqrt(compute_budget / 120)
    d_opt = 20 * n_opt
    return ChinchillaEstimate(
        gpu=gpu,
        days=days,
        theoretical_flops=theoretical_flops,
        compute_budget=compute_budget,
        n_opt=n_opt,
        d_opt=d_opt,
    )


def _count_data_files(data_dir: Path) -> int:
    """
    Count all data files under the immediate sub-directories of data_dir.

    Args:
        data_dir (Path): Directory whose sub-directories contain data files.

    Returns:
        int: Total number of files found across all sub-directories.
    """
    n = 0
    for ds_dir in data_dir.iterdir():
        if ds_dir.is_dir():
            for f in ds_dir.iterdir():
                if f.is_file():
                    n += 1
    return n


def estimate_dataset_gap(
    data_dir: Path,
    split_token_counts: SplitTokenCounts,
    d_opt: float,
    train_split_size: float,
) -> DatasetGapEstimate:
    """
    Estimate how many additional data files are needed to reach D_opt.

    Args:
        data_dir (Path): Resolved data directory (as returned by
            resolve_data_dir).
        split_token_counts (SplitTokenCounts): Current train/val token counts.
        d_opt (float): Chinchilla-optimal training token target.
        train_split_size (float): Fraction of tokens in each file assigned to
            training.

    Returns:
        DatasetGapEstimate: Dataclass with current file count, average tokens
            per file, and the number of additional files needed.
    """
    n_files = _count_data_files(data_dir)
    total_tokens = split_token_counts.train_tokens + split_token_counts.val_tokens
    avg_tokens_per_file = total_tokens / n_files if n_files > 0 else 0.0
    train_tokens_per_file = avg_tokens_per_file * train_split_size
    deficit = d_opt - split_token_counts.train_tokens
    files_needed = (
        max(0, math.ceil(deficit / train_tokens_per_file))
        if train_tokens_per_file > 0
        else 0
    )
    return DatasetGapEstimate(
        train_tokens=split_token_counts.train_tokens,
        val_tokens=split_token_counts.val_tokens,
        n_files=n_files,
        avg_tokens_per_file=avg_tokens_per_file,
        files_needed=files_needed,
    )


def _count_model_parameters(model_config: ModelConfig) -> int:
    """
    Count the total number of trainable parameters in the language model.

    Accounts for GQA (reduced K/V projection), tied input/output embeddings,
    SwiGLU FFN, and per-block RMSNorm gains. No bias parameters are counted
    because all linear layers use bias=False.

    Args:
        model_config (ModelConfig): Model architecture configuration.

    Returns:
        int: Total parameter count.
    """
    D = model_config.embedding_dim
    H_ff = model_config.resolved_ffn_hidden_dim
    kv_size = (D // model_config.n_heads) * model_config.n_kv_heads
    embedding_params = model_config.vocab_size * D  # tied with LM head, counted once
    final_norm_params = D
    per_block = (
        D                        # RMSNorm 1 gain
        + D * (D + 2 * kv_size)  # QKV projection (no bias)
        + D * D                  # output projection (no bias)
        + D                      # RMSNorm 2 gain
        + D * 2 * H_ff           # SwiGLU linear_12 (gate + up, no bias)
        + H_ff * D               # SwiGLU linear_out (no bias)
    )
    return embedding_params + final_norm_params + model_config.n_decoder_blocks * per_block


def _activation_bytes_per_batch_item(model_config: ModelConfig) -> int:
    """
    Estimate bf16 activation bytes per batch item stored for backprop.
    """
    D = model_config.embedding_dim
    T = model_config.sequence_length
    H_ff = model_config.resolved_ffn_hidden_dim
    kv_size = (D // model_config.n_heads) * model_config.n_kv_heads
    L = model_config.n_decoder_blocks
    V = model_config.vocab_size
    # Per layer: residuals (×2), norms (×2), Q/K/V projections, attn output, SwiGLU intermediates
    per_layer = 2 * (7 * D + 2 * kv_size + 3 * H_ff)  # 2 bytes for bf16
    return T * (
        2 * D * 2      # embedding output + final norm output (bf16)
        + L * per_layer
        + V * 2        # logits (bf16)
    )


def estimate_model_fit(
    model_config: ModelConfig,
    gpu: str,
    split_token_counts: SplitTokenCounts,
    batch_size: Optional[int] = None,
) -> ModelFitEstimate:
    """
    Estimate the maximum batch size that fits in GPU memory.

    When ``batch_size`` is provided it is used directly, skipping the memory
    capacity calculation.  Otherwise, accounts for static memory (parameters,
    gradients, AdamW states) and dynamic activation memory per batch item,
    leaving a 10 % headroom.

    Args:
        model_config (ModelConfig): Model architecture configuration.
        gpu (str): GPU model identifier used to look up HBM capacity.
        split_token_counts (SplitTokenCounts): Used to derive iteration counts
            from the fitted batch size.
        batch_size (int | None): User-supplied batch size override. When given,
            the GPU memory calculation is skipped.

    Returns:
        ModelFitEstimate: Dataclass with parameter count, max batch size,
            tokens per iteration, and iteration counts.
    """
    n_params = _count_model_parameters(model_config)
    if batch_size is not None:
        max_batch_size = batch_size
    else:
        gpu_memory = GPU_MEMORY_GiB[gpu] * (1024 ** 3)
        static_memory = n_params * BYTES_PER_PARAM
        activation_budget = gpu_memory * 0.9 - static_memory
        act_per_item = _activation_bytes_per_batch_item(model_config)
        max_batch_size = (
            max(1, int(activation_budget // act_per_item)) if activation_budget > 0 else 0
        )
    T = model_config.sequence_length
    tokens_per_iteration = max_batch_size * T
    return ModelFitEstimate(
        n_params=n_params,
        max_batch_size=max_batch_size,
        tokens_per_iteration=tokens_per_iteration,
        train_iterations=(
            split_token_counts.train_tokens // tokens_per_iteration
            if tokens_per_iteration > 0
            else 0
        ),
        val_iterations=(
            split_token_counts.val_tokens // tokens_per_iteration
            if tokens_per_iteration > 0
            else 0
        ),
    )


def estimate_token_budget(
    config: TrainingConfig,
    gpu: Optional[str] = None,
    days: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> TokenBudgetEstimate:
    """
    Estimate token counts, Chinchilla targets, and GPU fit for a training run.

    Args:
        config (TrainingConfig): Training configuration (data dir, model path,
            tokenizer, batch size, etc.).
        gpu (str | None): GPU model for Chinchilla and memory analysis. When
            None, those sections are skipped.
        days (float | None): Rental duration for Chinchilla analysis. When
            None, those sections are skipped.
        batch_size (int | None): User-supplied batch size override passed to
            ``estimate_model_fit``. When given, the GPU memory calculation is
            skipped.

    Returns:
        TokenBudgetEstimate: Dataclass combining token counts with optional
            Chinchilla, dataset-gap, and model-fit estimates.

    Raises:
        ValueError: If config validation fails or the data directory is
            invalid.
    """
    validate_config(config)
    data_dir = resolve_data_dir(config.data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"{config.data_dir} is not a valid data directory")

    model_config = ModelConfig.from_yaml(config.model_config)
    tokenizer = TiktokenTokenizer(config.tokenizer_encoding)
    split_token_counts = count_pre_training_tokens(
        data_path=data_dir,
        train_split_size=config.train_split_size,
        tokenizer=tokenizer,
        random_seed=config.seed,
    )
    tokens_per_iteration = config.batch_size * model_config.sequence_length

    chinchilla = None
    dataset_gap = None
    model_fit = None
    if gpu is not None and days is not None:
        chinchilla = estimate_chinchilla(gpu, days)
        dataset_gap = estimate_dataset_gap(
            data_dir, split_token_counts, chinchilla.d_opt, config.train_split_size
        )
        model_fit = estimate_model_fit(model_config, gpu, split_token_counts, batch_size=batch_size)

    return TokenBudgetEstimate(
        split_token_counts=split_token_counts,
        tokens_per_iteration=tokens_per_iteration,
        train_iterations=split_token_counts.train_tokens // tokens_per_iteration,
        val_iterations=split_token_counts.val_tokens // tokens_per_iteration,
        chinchilla=chinchilla,
        dataset_gap=dataset_gap,
        model_fit=model_fit,
    )


def format_token_budget_estimate(estimate: TokenBudgetEstimate) -> str:
    """
    Format a TokenBudgetEstimate as a human-readable multi-line string.

    When Chinchilla data is present, sections are printed in this order:
    Chinchilla-Optimal Compute Budget, Current Dataset, Model Analysis.
    Without Chinchilla data, only the basic token counts are printed.

    Args:
        estimate (TokenBudgetEstimate): The estimate to format.

    Returns:
        str: Multi-line formatted string ready for printing or writing to a
            file.
    """
    if estimate.chinchilla is None:
        return "\n".join([
            f"Train_tokens: {estimate.split_token_counts.train_tokens}",
            f"Validation tokens: {estimate.split_token_counts.val_tokens}",
            f"Tokens per iteration: {estimate.tokens_per_iteration}",
            f"Suggested max iterations: {estimate.train_iterations}",
            f"Validation iterations: {estimate.val_iterations}",
        ])

    c = estimate.chinchilla
    days_str = str(int(c.days)) if c.days == int(c.days) else str(c.days)

    lines = [
        "=== Chinchilla-Optimal Compute Budget ===",
        f"GPU:                        {c.gpu}",
        f"Rental duration:            {days_str} days",
        f"Theoretical FLOP/s:         {_fmt(c.theoretical_flops / 1e12)} TFLOP/s",
        f"MFU:                        45%",
        f"Real compute budget (C):    {_fmt(c.compute_budget)} FLOP",
        f"Optimal parameters (N_opt): {_fmt(c.n_opt)}",
        f"Optimal tokens (D_opt):     {_fmt(c.d_opt)}",
    ]

    dg = estimate.dataset_gap
    if dg is not None:
        if dg.files_needed == 0:
            files_str = f"{dg.n_files} current (sufficient for D_opt)"
        else:
            files_str = f"{dg.n_files} current, {dg.files_needed} more needed"
        lines += [
            "",
            "=== Current Dataset ===",
            f"Training tokens:            {dg.train_tokens / 1e9:.2f} B",
            f"Validation tokens:          {dg.val_tokens / 1e9:.2f} B",
            f"Avg tokens per file:        {_fmt(dg.avg_tokens_per_file)}",
            f"Files to reach D_opt:       {files_str}",
        ]

    mf = estimate.model_fit
    if mf is not None:
        lines += [
            "",
            "=== Model Analysis ===",
            f"Current model parameters:   {mf.n_params / 1e6:.2f} M",
            f"Max batch size (GPU fit):   {mf.max_batch_size}",
            f"Tokens per iteration:       {mf.tokens_per_iteration}",
            f"Suggested max iterations:   {mf.train_iterations}",
            f"Validation iterations:      {mf.val_iterations}",
        ]

    return "\n".join(lines)


def print_token_budget_estimate(
    config: TrainingConfig,
    gpu: Optional[str] = None,
    days: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> None:
    """
    Print the formatted token budget estimate to stdout.

    Args:
        config (TrainingConfig): Training configuration.
        gpu (str | None): Optional GPU model for Chinchilla analysis.
        days (float | None): Optional rental duration in days.
        batch_size (int | None): Optional user-supplied batch size override.
    """
    print(format_token_budget_estimate(
        estimate_token_budget(config, gpu=gpu, days=days, batch_size=batch_size)
    ))
