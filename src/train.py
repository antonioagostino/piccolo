from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from src.dataset import PreTrainingDataset
from src.tokenizer import TiktokenTokenizer
from src.transformer import (
    LanguageModel,
    ModelConfig,
    compile_language_model,
    get_supported_weights_precision,
    language_model_loss,
    training_step,
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
    train_split_size: float
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    lr_warmup_iterations: int
    max_iterations: int | None
    weight_decay: float
    max_grad_norm: float | None
    device: str
    compile_model: bool
    seed: int
    log_every_tokens: int
    wandb: WandbConfig


class MetricsLogger:
    def log(self, metrics: dict[str, float | int], step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbMetricsLogger(MetricsLogger):
    def __init__(self,
                 config: WandbConfig,
                 training_config: TrainingConfig,
                 model_config: ModelConfig) -> None:
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
            config={
                "training": config_to_dict(training_config),
                "model": model_config.__dict__,
            },
        )

    def log(self, metrics: dict[str, float | int], step: int) -> None:
        self.__wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.__run is not None:
            self.__run.finish()


def config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return {
        "data_dir": str(config.data_dir),
        "model_config": str(config.model_config),
        "tokenizer_encoding": config.tokenizer_encoding,
        "train_split_size": config.train_split_size,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "lr_warmup_iterations": config.lr_warmup_iterations,
        "max_iterations": config.max_iterations,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "device": config.device,
        "compile_model": config.compile_model,
        "seed": config.seed,
        "log_every_tokens": config.log_every_tokens,
        "wandb": {
            "enabled": config.wandb.enabled,
            "project": config.wandb.project,
            "run_name": config.wandb.run_name,
            "mode": config.wandb.mode,
        },
    }


def as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def load_training_config(config_path: Path) -> TrainingConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    root = as_mapping(raw_config, "The training config file")
    training = as_mapping(root.get("training", root), "training")
    wandb_config = as_mapping(training.get("wandb", {}), "training.wandb")

    return TrainingConfig(
        data_dir=Path(training.get("data_dir", "./data")),
        model_config=Path(training.get("model_config", "./configs/model.yaml")),
        tokenizer_encoding=str(training.get("tokenizer_encoding", "cl100k_base")),
        train_split_size=float(training.get("train_split_size", 0.9)),
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
        seed=int(training.get("seed", 42)),
        log_every_tokens=int(training.get("log_every_tokens", 2048)),
        wandb=WandbConfig(
            enabled=bool(wandb_config.get("enabled", True)),
            project=str(wandb_config.get("project", "friendbots")),
            run_name=wandb_config.get("run_name"),
            mode=str(wandb_config.get("mode", "online")),
        ),
    )


def validate_config(config: TrainingConfig) -> None:
    if not 0.0 < config.train_split_size <= 1.0:
        raise ValueError("train_split_size must be in the interval (0, 1]")
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


def resolve_data_dir(data_dir: Path) -> Path:
    if (data_dir / "raw_text").is_dir():
        return data_dir / "raw_text"
    return data_dir


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_learning_rate(config: TrainingConfig, iteration: int) -> float:
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
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def build_logger(config: TrainingConfig, model_config: ModelConfig) -> MetricsLogger:
    if config.wandb.enabled:
        return WandbMetricsLogger(config.wandb, config, model_config)
    return MetricsLogger()


def reset_inference_state(language_model: torch.nn.Module) -> None:
    model = getattr(language_model, "_orig_mod", language_model)
    transformer_decoder = getattr(model, "transformer_decoder", None)
    if transformer_decoder is None:
        return

    transformer_decoder.global_token_counter = 0
    transformer_decoder.kv_cache.clear()


def validate(
    language_model: torch.nn.Module,
    pre_training_dataset: PreTrainingDataset,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float | None:
    language_model.eval()
    total_token_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        progress = tqdm(desc="validation", unit="token")
        while True:
            try:
                inputs, targets = pre_training_dataset.get_sequential_batch("val")
            except StopIteration:
                break

            reset_inference_state(language_model)
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = language_model(inputs)
                loss = language_model_loss(logits, targets)

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


def train(config: TrainingConfig) -> None:
    validate_config(config)
    seed_everything(config.seed)

    data_dir = resolve_data_dir(config.data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"{config.data_dir} is not a valid data directory")

    device = resolve_device(config.device)
    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and amp_dtype == torch.float16

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
    ).to(device)
    language_model = compile_language_model(language_model, enabled=config.compile_model)

    optimizer = torch.optim.AdamW(
        language_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)
    logger = build_logger(config, model_config)

    optimizer_steps = 0
    train_tokens_seen = 0
    train_token_loss = 0.0
    next_log_tokens = config.log_every_tokens
    try:
        pre_training_dataset = PreTrainingDataset(
            str(data_dir),
            model_config.sequence_length,
            config.train_split_size,
            config.batch_size,
            tokenizer,
            device,
            random_seed=config.seed,
        )
        progress = tqdm(desc="training", unit="token")

        while True:
            try:
                inputs, targets = pre_training_dataset.get_sequential_batch("train")
            except StopIteration:
                break
            if config.max_iterations is not None and optimizer_steps >= config.max_iterations:
                break

            tokens_in_batch = inputs.numel()
            optimizer_steps += 1
            learning_rate = get_learning_rate(config, optimizer_steps)
            set_optimizer_learning_rate(optimizer, learning_rate)
            loss = training_step(
                language_model=language_model,
                optimizer=optimizer,
                scaler=scaler,
                inputs=inputs,
                targets=targets,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                max_grad_norm=config.max_grad_norm,
            )
            train_tokens_seen += tokens_in_batch
            train_token_loss += loss * tokens_in_batch
            avg_train_loss = train_token_loss / train_tokens_seen
            progress.update(tokens_in_batch)
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

        val_loss = validate(
            language_model=language_model,
            pre_training_dataset=pre_training_dataset,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
        )
        if val_loss is not None:
            logger.log(
                {
                    "val/loss": val_loss,
                    "train/tokens_seen": train_tokens_seen,
                },
                step=max(train_tokens_seen, 1),
            )
    finally:
        logger.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the friendsbot language model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_training_config(args.config))
