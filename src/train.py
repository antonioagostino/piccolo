from __future__ import annotations

import argparse
import dataclasses
import math
import random
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from src.dataset import TokenizedDataset
from src.tokenizer import TiktokenTokenizer
from src.transformer import (
    LanguageModel,
    ModelConfig,
    compile_language_model,
    forward_backward_micro_step,
    get_supported_weights_precision,
    optimizer_step,
)


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    run_name: str | None
    mode: str


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    model_config: Path
    tokenizer_encoding: str
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    lr_warmup_iterations: int
    max_iterations: int | None
    weight_decay: float
    max_grad_norm: float | None
    device: str
    compile_model: bool
    gradient_checkpointing: bool
    gradient_accumulation_steps: int
    seed: int
    log_every_tokens: int
    val_every_iterations: int
    val_max_iterations: int | None
    checkpoint_dir: Path
    resume_from: Path | None
    wandb: WandbConfig


class MetricsLogger:
    """
    No-op metrics logger used as a default when W&B is disabled.
    """

    @property
    def run_id(self) -> str | None:
        return None

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        """
        Record a dictionary of metrics at a given training step.

        Args:
            metrics (dict[str, float | int]): Mapping of metric names to
                values.
            step (int): Global step index (e.g. tokens seen).
        """
        pass

    def finish(self) -> None:
        """
        Finalise the logging session and flush any pending data."""
        pass


class WandbMetricsLogger(MetricsLogger):
    """
    Metrics logger that streams training metrics to Weights & Biases.
    """

    def __init__(self,
                 config: WandbConfig,
                 training_config: TrainingConfig,
                 model_config: ModelConfig,
                 resume_run_id: str | None = None) -> None:
        """
        Initialise and start a W&B run.

        Args:
            config (WandbConfig): W&B project, run name, and mode settings.
            training_config (TrainingConfig): Full training configuration
                logged as run metadata.
            model_config (ModelConfig): Model architecture configuration
                logged as run metadata.
            resume_run_id (str | None): W&B run ID to resume. When set,
                metrics are appended to the existing run instead of starting
                a new one.

        Raises:
            RuntimeError: If the ``wandb`` package is not installed.
        """
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but wandb is not installed. "
                "Install requirements.txt or set training.wandb.enabled=false."
            ) from exc

        self.__wandb: Any = wandb
        self.__run: Any = self.__wandb.init(
            project=config.project,
            name=config.run_name,
            mode=config.mode,
            id=resume_run_id,
            resume="must" if resume_run_id is not None else None,
            config={
                "training": config_to_dict(training_config),
                "model": model_config.__dict__,
            },
        )
        # When resuming, skip any step already present in the run so we never
        # log backwards and trigger a step-ordering error from W&B.
        self._skip_until_step: int = 0
        if resume_run_id is not None and self.__run is not None:
            self._skip_until_step = int(self.__run.summary.get("train/tokens_seen", 0))

    @property
    def run_id(self) -> str | None:
        return self.__run.id if self.__run is not None else None

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        """
        Log a dictionary of metrics to the active W&B run.

        Args:
            metrics (dict[str, float | int]): Mapping of metric names to
                values.
            step (int): Global step index used as the x-axis in W&B charts.
        """
        if step <= self._skip_until_step:
            return
        self.__wandb.log(metrics, step=step)

    def finish(self) -> None:
        """
        Finalise the W&B run, uploading any remaining queued data."""
        if self.__run is not None:
            self.__run.finish()


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    """
    Serialise a TrainingConfig to a plain dict suitable for JSON / W&B logging.

    Args:
        config (TrainingConfig): The training configuration to serialise.

    Returns:
        dict[str, Any]: Flat dictionary representation of the config.
    """
    return {
        "data_dir": str(config.data_dir),
        "model_config": str(config.model_config),
        "tokenizer_encoding": config.tokenizer_encoding,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "lr_warmup_iterations": config.lr_warmup_iterations,
        "max_iterations": config.max_iterations,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "device": config.device,
        "compile_model": config.compile_model,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "seed": config.seed,
        "log_every_tokens": config.log_every_tokens,
        "val_every_iterations": config.val_every_iterations,
        "val_max_iterations": config.val_max_iterations,
        "checkpoint_dir": str(config.checkpoint_dir),
        "resume_from": str(config.resume_from) if config.resume_from is not None else None,
        "wandb": {
            "enabled": config.wandb.enabled,
            "project": config.wandb.project,
            "run_name": config.wandb.run_name,
            "mode": config.wandb.mode,
        },
    }


def as_mapping(value: Any, name: str) -> dict[str, Any]:
    """
    Assert that a YAML-parsed value is a dict and return it.

    Args:
        value (Any): The value to check.
        name (str): Human-readable name of the value, used in the error
            message.

    Returns:
        dict[str, Any]: The value cast to a dict.

    Raises:
        ValueError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def load_training_config(config_path: Path) -> TrainingConfig:
    """
    Parse and return a TrainingConfig from a YAML file.

    Missing keys fall back to their documented defaults.

    Args:
        config_path (Path): Path to the YAML training configuration file.

    Returns:
        TrainingConfig: The parsed training configuration.
    """
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    root = as_mapping(raw_config, "The training config file")
    training = as_mapping(root.get("training", root), "training")
    wandb_config = as_mapping(training.get("wandb", {}), "training.wandb")

    return TrainingConfig(
        data_dir=Path(training.get("data_dir", "./data/tokenized")),
        model_config=Path(training.get("model_config", "./configs/model.yaml")),
        tokenizer_encoding=str(training.get("tokenizer_encoding", "cl100k_base")),
        batch_size=int(training.get("batch_size", 4)),
        learning_rate=float(training.get("learning_rate", 3e-4)),
        min_learning_rate=float(training.get("min_learning_rate", 3e-5)),
        lr_warmup_iterations=int(training.get("lr_warmup_iterations", 0)),
        max_iterations=(
            None
            if training.get("max_iterations") is None
            else int(training["max_iterations"])
        ),
        weight_decay=float(training.get("weight_decay", 0.01)),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        device=str(training.get("device", "auto")),
        compile_model=bool(training.get("compile_model", False)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        seed=int(training.get("seed", 42)),
        log_every_tokens=int(training.get("log_every_tokens", 2048)),
        val_every_iterations=int(training.get("val_every_iterations", 500)),
        val_max_iterations=(
            None
            if training.get("val_max_iterations") is None
            else int(training["val_max_iterations"])
        ),
        checkpoint_dir=Path(training.get("checkpoint_dir", "./checkpoints")),
        resume_from=(
            Path(training["resume_from"])
            if training.get("resume_from") is not None
            else None
        ),
        wandb=WandbConfig(
            enabled=bool(wandb_config.get("enabled", True)),
            project=str(wandb_config.get("project", "friendsbot")),
            run_name=wandb_config.get("run_name"),
            mode=str(wandb_config.get("mode", "online")),
        ),
    )


def validate_config(config: TrainingConfig) -> None:
    """
    Raise if any field in the training config violates its constraint.

    Args:
        config (TrainingConfig): The configuration to validate.

    Raises:
        ValueError: If any field is out of range or logically inconsistent.
    """
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be greater than or equal to 0")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")
    if config.lr_warmup_iterations < 0:
        raise ValueError("lr_warmup_iterations must be greater than or equal to 0")
    if config.max_iterations is not None and config.max_iterations <= 0:
        raise ValueError("max_iterations must be greater than 0 when configured")
    if config.max_iterations is not None and config.max_iterations <= config.lr_warmup_iterations:
        raise ValueError("max_iterations must be greater than lr_warmup_iterations")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be greater than or equal to 0")
    if config.max_grad_norm is not None and config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be greater than 0 when configured")
    if config.log_every_tokens <= 0:
        raise ValueError("log_every_tokens must be greater than 0")
    if config.val_every_iterations <= 0:
        raise ValueError("val_every_iterations must be greater than 0")
    if config.val_max_iterations is not None and config.val_max_iterations <= 0:
        raise ValueError("val_max_iterations must be greater than 0 when configured")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.resume_from is not None and not config.resume_from.is_file():
        raise ValueError(f"resume_from path does not exist: {config.resume_from}")


def resolve_device(requested_device: str) -> torch.device:
    """
    Resolve a device string to a torch.device.

    Args:
        requested_device (str): Device string from the training config.
            Pass ``"auto"`` to automatically select CUDA when available.

    Returns:
        torch.device: The resolved compute device.

    Raises:
        ValueError: If ``"cuda"`` is requested but CUDA is not available.
    """
    if requested_device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    return device


def seed_everything(seed: int) -> None:
    """
    Set all relevant random seeds for reproducible training.

    Args:
        seed (int): The integer seed applied to Python, PyTorch, and CUDA RNGs.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_learning_rate(config: TrainingConfig, iteration: int) -> float:
    """
    Compute the learning rate for a given training iteration.

    Applies linear warm-up from 0 to ``learning_rate`` over
    ``lr_warmup_iterations`` steps, then cosine decay to
    ``min_learning_rate`` over the remaining iterations.

    Args:
        config (TrainingConfig): Training configuration holding the LR
            schedule parameters.
        iteration (int): Current training step (1-indexed).

    Returns:
        float: The learning rate to use at this iteration.

    Raises:
        ValueError: If iteration is not positive.
    """
    if iteration <= 0:
        raise ValueError("iteration must be greater than 0")

    if config.lr_warmup_iterations > 0 and iteration <= config.lr_warmup_iterations:
        return config.learning_rate * iteration / config.lr_warmup_iterations

    if config.max_iterations is None:
        return config.learning_rate

    decay_iterations = config.max_iterations - config.lr_warmup_iterations
    decay_iteration = max(iteration - config.lr_warmup_iterations - 1, 0)
    decay_progress = min(decay_iteration / decay_iterations, 1.0)
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return config.min_learning_rate + cosine_multiplier * (
        config.learning_rate - config.min_learning_rate
    )


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate in all optimizer parameter groups.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update.
        learning_rate (float): New learning rate value.
    """
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def build_logger(
    config: TrainingConfig,
    model_config: ModelConfig,
    resume_run_id: str | None = None,
) -> MetricsLogger:
    """
    Construct the appropriate metrics logger based on the training config.

    Args:
        config (TrainingConfig): Training configuration; inspects
            ``config.wandb.enabled`` to choose the logger type.
        model_config (ModelConfig): Model configuration passed to the W&B
            logger for metadata logging.
        resume_run_id (str | None): W&B run ID to resume. Forwarded to
            WandbMetricsLogger when W&B is enabled.

    Returns:
        MetricsLogger: A WandbMetricsLogger if W&B is enabled, otherwise the
            no-op MetricsLogger.
    """
    if config.wandb.enabled:
        return WandbMetricsLogger(config.wandb, config, model_config, resume_run_id)
    return MetricsLogger()


def reset_inference_state(language_model: torch.nn.Module) -> None:
    """
    Reset the KV-cache and token counter on the transformer decoder.

    Called between validation sequences to prevent context bleed across
    independent sequences during inference. Handles compiled models
    (``_orig_mod``) transparently.

    Args:
        language_model (torch.nn.Module): The language model whose internal
            decoder state should be reset.
    """
    model = getattr(language_model, "_orig_mod", language_model)
    transformer_decoder = getattr(model, "transformer_decoder", None)
    if transformer_decoder is None:
        return

    transformer_decoder.global_token_counter = 0
    transformer_decoder.kv_cache.clear()


def load_checkpoint(
    path: Path,
    language_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int, float, float]:
    """
    Load a training checkpoint and restore model, optimizer, and scaler state.

    Args:
        path (Path): Path to the checkpoint file.
        language_model (torch.nn.Module): Model to restore weights into.
        optimizer (torch.optim.Optimizer): Optimizer to restore state into.
        scaler (torch.amp.GradScaler): Grad scaler to restore state into.

    Returns:
        tuple[int, int, float, float]: ``(optimizer_steps, train_tokens_seen,
            best_val_loss, train_token_loss)`` read from the checkpoint.
    """
    checkpoint = torch.load(path, weights_only=True)
    model = getattr(language_model, "_orig_mod", language_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return (
        checkpoint["optimizer_steps"],
        checkpoint["train_tokens_seen"],
        checkpoint["val_loss"],
        checkpoint.get("train_token_loss", 0.0),
        checkpoint.get("wandb_run_id"),
    )


def save_checkpoint(
    path: Path,
    language_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    optimizer_steps: int,
    train_tokens_seen: int,
    train_token_loss: float,
    val_loss: float,
    wandb_run_id: str | None = None,
) -> None:
    """
    Save a training checkpoint to disk.

    Unwraps compiled models (``_orig_mod``) before saving so the checkpoint
    contains plain module state and can be loaded without torch.compile.

    Args:
        path (Path): Destination file path; parent directories are created
            automatically.
        language_model (torch.nn.Module): The model whose parameters are saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state is saved.
        scaler (torch.amp.GradScaler): The gradient scaler whose state is saved.
        optimizer_steps (int): Number of optimizer steps completed.
        train_tokens_seen (int): Total training tokens processed so far.
        train_token_loss (float): Cumulative sum of (loss * tokens) used to
            compute the running average training loss.
        val_loss (float): Validation loss at this checkpoint; use
            ``float("nan")`` when no validation was run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model = getattr(language_model, "_orig_mod", language_model)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "optimizer_steps": optimizer_steps,
            "train_tokens_seen": train_tokens_seen,
            "train_token_loss": train_token_loss,
            "val_loss": val_loss,
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def validate(
    language_model: torch.nn.Module,
    dataset: TokenizedDataset,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_iterations: int | None = None,
) -> float | None:
    """
    Run a validation pass and return the mean per-token loss.

    Args:
        language_model (torch.nn.Module): The model to evaluate (set to eval
            mode internally).
        dataset (TokenizedDataset): Dataset whose validation split is read.
        device (torch.device): Device that tensors are moved to.
        amp_dtype (torch.dtype): dtype used inside the autocast region.
        use_amp (bool): Whether to enable automatic mixed precision.
        max_iterations (int | None): Maximum number of batches to evaluate.
            When None the validation split is consumed fully.

    Returns:
        float | None: Mean per-token cross-entropy loss over the evaluated
            batches, or None if no validation batches were produced.
    """
    dataset.reset_split("val")
    language_model.eval()
    total_token_loss = 0.0
    total_tokens = 0
    iteration = 0

    with torch.no_grad():
        progress = tqdm(
            desc="\033[1mValidation\033[0m",
            unit=" total tokens",
            bar_format="{desc}: {n_fmt}{unit} [elapsed: {elapsed}, {rate_fmt}{postfix}]",
        )
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break
            try:
                inputs, targets = dataset.get_sequential_batch("val")
            except StopIteration:
                break
            iteration += 1

            reset_inference_state(language_model)
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = language_model(inputs, targets=targets)

            tokens_in_batch = inputs.numel()
            total_tokens += tokens_in_batch
            total_token_loss += float(loss.detach().item()) * tokens_in_batch
            progress.update(tokens_in_batch)
            progress.set_postfix(loss=f"{total_token_loss / total_tokens:.4f}")

        progress.close()

    reset_inference_state(language_model)
    if total_tokens == 0:
        return None

    return total_token_loss / total_tokens


def train(config: TrainingConfig, wandb_resume_id: str | None = None) -> None:
    """Run the full pre-training loop from a TrainingConfig.

    Validates the config, initialises the model, optimizer, and dataset,
    then iterates over sequential batches. Logs metrics periodically and
    runs a full validation pass at the end.

    Args:
        config (TrainingConfig): Complete training configuration.

    Raises:
        ValueError: If config validation fails or the data directory is
            missing.
    """
    validate_config(config)
    seed_everything(config.seed)

    data_dir = config.data_dir
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a valid data directory")

    device = resolve_device(config.device)
    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    torch.set_float32_matmul_precision("high")

    model_config = ModelConfig.from_yaml(config.model_config)
    tokenizer = TiktokenTokenizer(config.tokenizer_encoding)
    tokenizer_vocab_size = tokenizer.tokenizer.n_vocab
    if model_config.vocab_size < tokenizer_vocab_size:
        raise ValueError(
            "model.vocab_size must be greater than or equal to the tokenizer "
            f"vocabulary size ({tokenizer_vocab_size})"
        )

    language_model: torch.nn.Module = LanguageModel.from_config(
        model_config,
        kv_cache={},
        device=device,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    language_model = compile_language_model(language_model, enabled=config.compile_model)

    optimizer = torch.optim.AdamW(
        language_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)

    optimizer_steps = 0
    train_tokens_seen = 0
    train_token_loss = 0.0
    next_log_tokens = config.log_every_tokens
    best_val_loss = float("inf")
    wandb_run_id: str | None = None

    if config.resume_from is not None:
        optimizer_steps, train_tokens_seen, best_val_loss, train_token_loss, wandb_run_id = load_checkpoint(
            config.resume_from, language_model, optimizer, scaler
        )
        next_log_tokens = (train_tokens_seen // config.log_every_tokens + 1) * config.log_every_tokens
        set_optimizer_learning_rate(optimizer, get_learning_rate(config, optimizer_steps))

    logger = build_logger(config, model_config, resume_run_id=wandb_resume_id or wandb_run_id)

    try:
        dataset = TokenizedDataset(
            data_dir=data_dir,
            sequence_length=model_config.sequence_length,
            batch_size=config.batch_size,
        )
        if config.resume_from is not None:
            dataset._offset["train"] = train_tokens_seen

        def _run_validation() -> float | None:
            result = validate(
                language_model=language_model,
                dataset=dataset,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                max_iterations=config.val_max_iterations,
            )

            language_model.train()
            return result

        progress = tqdm(
            desc="\033[1mTraining\033[0m",
            unit=" total tokens",
            bar_format="{desc}: {n_fmt}{unit} [elapsed: {elapsed}, {rate_fmt}{postfix}]",
            initial=train_tokens_seen,
        )

        while True:
            if config.max_iterations is not None and optimizer_steps >= config.max_iterations:
                break

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            tokens_in_optimizer_step = 0

            for _ in range(config.gradient_accumulation_steps):
                try:
                    inputs, targets = dataset.get_sequential_batch("train")
                except StopIteration:
                    break
                tokens_in_micro_batch = inputs.numel()
                micro_loss = forward_backward_micro_step(
                    language_model=language_model,
                    scaler=scaler,
                    inputs=inputs,
                    targets=targets,
                    device=device,
                    amp_dtype=amp_dtype,
                    use_amp=use_amp,
                    loss_scale=1.0 / config.gradient_accumulation_steps,
                )
                accumulated_loss += micro_loss
                tokens_in_optimizer_step += tokens_in_micro_batch

            if tokens_in_optimizer_step == 0:
                break

            optimizer_steps += 1
            learning_rate = get_learning_rate(config, optimizer_steps)
            set_optimizer_learning_rate(optimizer, learning_rate)
            optimizer_step(
                language_model=language_model,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=config.max_grad_norm,
            )

            loss = accumulated_loss / config.gradient_accumulation_steps
            train_tokens_seen += tokens_in_optimizer_step
            train_token_loss += loss * tokens_in_optimizer_step
            avg_train_loss = train_token_loss / train_tokens_seen
            progress.update(tokens_in_optimizer_step)
            progress.set_postfix(loss=f"{avg_train_loss:.4f}", lr=f"{learning_rate:.2e}")

            if train_tokens_seen >= next_log_tokens:
                logger.log(
                    {
                        "train/loss": loss,
                        "train/avg_loss": avg_train_loss,
                        "train/tokens_seen": train_tokens_seen,
                        "train/optimizer_steps": optimizer_steps,
                        "train/learning_rate": learning_rate,
                    },
                    step=train_tokens_seen,
                )
                while next_log_tokens <= train_tokens_seen:
                    next_log_tokens += config.log_every_tokens

            if optimizer_steps % config.val_every_iterations == 0:
                step_val_loss = _run_validation()
                if step_val_loss is not None:
                    logger.log(
                        {
                            "val/loss": step_val_loss,
                            "train/tokens_seen": train_tokens_seen,
                            "train/optimizer_steps": optimizer_steps,
                        },
                        step=train_tokens_seen,
                    )
                    save_checkpoint(
                        path=config.checkpoint_dir / "checkpoint_latest.pt",
                        language_model=language_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        optimizer_steps=optimizer_steps,
                        train_tokens_seen=train_tokens_seen,
                        train_token_loss=train_token_loss,
                        val_loss=step_val_loss,
                        wandb_run_id=logger.run_id,
                    )
                    if step_val_loss < best_val_loss:
                        best_val_loss = step_val_loss
                        save_checkpoint(
                            path=config.checkpoint_dir / "checkpoint_best.pt",
                            language_model=language_model,
                            optimizer=optimizer,
                            scaler=scaler,
                            optimizer_steps=optimizer_steps,
                            train_tokens_seen=train_tokens_seen,
                            train_token_loss=train_token_loss,
                            val_loss=step_val_loss,
                            wandb_run_id=logger.run_id,
                        )

        progress.close()

        if train_tokens_seen > 0:
            logger.log(
                {
                    "train/final_loss": train_token_loss / train_tokens_seen,
                    "train/tokens_seen": train_tokens_seen,
                    "train/optimizer_steps": optimizer_steps,
                    "train/learning_rate": get_learning_rate(config, optimizer_steps),
                },
                step=train_tokens_seen,
            )

        final_val_loss = _run_validation()
        if final_val_loss is not None:
            logger.log(
                {
                    "val/loss": final_val_loss,
                    "train/tokens_seen": train_tokens_seen,
                    "train/optimizer_steps": optimizer_steps,
                },
                step=max(train_tokens_seen, 1),
            )

        save_checkpoint(
            path=config.checkpoint_dir / "checkpoint_last.pt",
            language_model=language_model,
            optimizer=optimizer,
            scaler=scaler,
            optimizer_steps=optimizer_steps,
            train_tokens_seen=train_tokens_seen,
            train_token_loss=train_token_loss,
            val_loss=final_val_loss if final_val_loss is not None else float("nan"),
            wandb_run_id=logger.run_id,
        )
    finally:
        logger.finish()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training entry point.

    Returns:
        argparse.Namespace: Parsed arguments with a ``config`` field holding
            the path to the YAML training configuration.
    """
    parser = argparse.ArgumentParser(description="Train the friendsbot language model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--wandb-resume-id",
        type=str,
        default=None,
        help="W&B run ID to resume (overrides the run ID stored in the checkpoint).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_training_config(args.config)
    if args.resume_from is not None:
        config = dataclasses.replace(config, resume_from=args.resume_from)
    train(config, wandb_resume_id=args.wandb_resume_id)
