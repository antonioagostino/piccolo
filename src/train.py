import argparse
import math
import random
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from tqdm.auto import tqdm  # type: ignore[import-untyped]
import wandb

from src.dataset import TokenizedPreTrainingDataset, TokenizedFinetuneDataset
from src.tokenizer import TiktokenTokenizer
from src.transformer import (
    LanguageModel,
    TransformerDecoder,
    get_supported_weights_precision,
)


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Parse config YAML file and return a dictionary of training config variables."""
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    assert isinstance(raw_config, dict), "Invalid YAML mapping"
    training_config = raw_config["training"]
    assert isinstance(training_config, dict), "Invalid YAML mapping"
    wandb_config = training_config["wandb"]
    assert isinstance(wandb_config, dict), "Invalid YAML mapping"

    config: dict[str, Any] = {}

    if (_v := training_config.get("data_dir", None)) is None:
        raise ValueError("Missing 'data_dir' in training config file")
    config["data_dir"] = Path(_v)

    if (_v := training_config.get("model_config", None)) is None:
        raise ValueError("Missing 'model_config' in training config file")
    config["model_config"] = Path(_v)

    if (_v := training_config.get("tokenizer_encoding", None)) is None:
        raise ValueError("Missing 'tokenizer_encoding' in training config file")
    config["tokenizer_encoding"] = str(_v)

    if (_v := training_config.get("batch_size", None)) is None:
        raise ValueError("Missing 'batch_size' in training config file")
    config["batch_size"] = int(_v)

    if (_v := training_config.get("learning_rate", None)) is None:
        raise ValueError("Missing 'learning_rate' in training config file")
    config["learning_rate"] = float(_v)

    if (_v := training_config.get("min_learning_rate", None)) is None:
        raise ValueError("Missing 'min_learning_rate' in training config file")
    config["min_learning_rate"] = float(_v)

    if (_v := training_config.get("lr_warmup_iterations", None)) is None:
        raise ValueError("Missing 'lr_warmup_iterations' in training config file")
    config["lr_warmup_iterations"] = int(_v)

    _v = training_config.get("max_iterations", None)
    config["max_iterations"] = None if _v is None else int(_v)

    if (_v := training_config.get("weight_decay", None)) is None:
        raise ValueError("Missing 'weight_decay' in training config file")
    config["weight_decay"] = float(_v)

    config["max_grad_norm"] = training_config.get("max_grad_norm", None)

    if (_v := training_config.get("device", None)) is None:
        raise ValueError("Missing 'device' in training config file")
    config["device"] = str(_v)

    if (_v := training_config.get("compile_model", None)) is None:
        raise ValueError("Missing 'compile_model' in training config file")
    config["compile_model"] = bool(_v)

    if (_v := training_config.get("gradient_checkpointing", None)) is None:
        raise ValueError("Missing 'gradient_checkpointing' in training config file")
    config["gradient_checkpointing"] = bool(_v)

    if (_v := training_config.get("gradient_accumulation_steps", None)) is None:
        raise ValueError("Missing 'gradient_accumulation_steps' in training config file")
    config["gradient_accumulation_steps"] = int(_v)

    if (_v := training_config.get("seed", None)) is None:
        raise ValueError("Missing 'seed' in training config file")
    config["seed"] = int(_v)

    if (_v := training_config.get("log_every_tokens", None)) is None:
        raise ValueError("Missing 'log_every_tokens' in training config file")
    config["log_every_tokens"] = int(_v)

    if (_v := training_config.get("val_every_iterations", None)) is None:
        raise ValueError("Missing 'val_every_iterations' in training config file")
    config["val_every_iterations"] = int(_v)

    _v = training_config.get("val_max_iterations", None)
    config["val_max_iterations"] = None if _v is None else int(_v)

    if (_v := training_config.get("n_epochs", None)) is None:
        raise ValueError("Missing 'n_epochs' in training config file")
    config["n_epochs"] = int(_v)

    if (_v := training_config.get("checkpoint_dir", None)) is None:
        raise ValueError("Missing 'checkpoint_dir' in training config file")
    config["checkpoint_dir"] = Path(_v)

    _v = training_config.get("init_from", None)
    config["init_from"] = Path(_v) if _v is not None else None

    _v = training_config.get("resume_from", None)
    config["resume_from"] = Path(_v) if _v is not None else None

    if (_v := wandb_config.get("enabled", None)) is None:
        raise ValueError("Missing 'enabled' in wandb config")
    config["wandb_enabled"] = bool(_v)

    if config["wandb_enabled"]:
        if (_v := wandb_config.get("project", None)) is None:
            raise ValueError("Missing 'project' in wandb config")
        config["wandb_project"] = str(_v)
        config["wandb_run_name"] = wandb_config.get("run_name", None)
        if (_v := wandb_config.get("mode", None)) is None:
            raise ValueError("Missing 'mode' in wandb config")
        config["wandb_mode"] = str(_v)
        config["wandb_resume_id"] = wandb_config.get("resume_id", None)
    else:
        config["wandb_project"] = None
        config["wandb_run_name"] = None
        config["wandb_mode"] = None
        config["wandb_resume_id"] = None

    return config


def training_config_to_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Serialise training config dictionary to a plain dict suitable for W&B logging."""
    return {
        "data_dir": str(config["data_dir"]),
        "model_config": str(config["model_config"]),
        "tokenizer_encoding": config["tokenizer_encoding"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "min_learning_rate": config["min_learning_rate"],
        "lr_warmup_iterations": config["lr_warmup_iterations"],
        "max_iterations": config["max_iterations"],
        "weight_decay": config["weight_decay"],
        "max_grad_norm": config["max_grad_norm"],
        "device": config["device"],
        "compile_model": config["compile_model"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "seed": config["seed"],
        "log_every_tokens": config["log_every_tokens"],
        "val_every_iterations": config["val_every_iterations"],
        "val_max_iterations": config["val_max_iterations"],
        "n_epochs": config["n_epochs"],
        "checkpoint_dir": str(config["checkpoint_dir"]),
        "init_from": str(config["init_from"]) if config["init_from"] is not None else None,
        "resume_from": str(config["resume_from"]) if config["resume_from"] is not None else None,
        "wandb": {
            "enabled": config["wandb_enabled"],
            "project": config["wandb_project"],
            "run_name": config["wandb_run_name"],
            "mode": config["wandb_mode"],
        },
    }


def model_config_to_dict(language_model: LanguageModel) -> dict[str, int | float | None]:
    """Serialize input LanguageModel config to a dictionary."""
    return {
        "vocab_size": language_model.vocab_size,
        "sequence_length": language_model.sequence_length,
        "embedding_dim": language_model.embedding_dim,
        "n_decoder_blocks": language_model.n_decoder_blocks,
        "n_heads": language_model.n_heads,
        "n_kv_heads": language_model.n_kv_heads,
        "ffn_hidden_dim": language_model.ffn_hidden_dim,
        "dropout_rate": language_model.dropout_rate,
    }


def validate_config(config: dict[str, Any]) -> None:
    """Raise if any value in the training config dict violates its constraint."""
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be greater than 0")
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if config["min_learning_rate"] < 0:
        raise ValueError("min_learning_rate must be greater than or equal to 0")
    if config["min_learning_rate"] > config["learning_rate"]:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")
    if config["lr_warmup_iterations"] < 0:
        raise ValueError("lr_warmup_iterations must be greater than or equal to 0")
    if config["max_iterations"] is not None and config["max_iterations"] <= 0:
        raise ValueError("max_iterations must be greater than 0 when configured")
    if config["max_iterations"] is not None and config["max_iterations"] <= config["lr_warmup_iterations"]:
        raise ValueError("max_iterations must be greater than lr_warmup_iterations")
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be greater than or equal to 0")
    if config["max_grad_norm"] is not None and config["max_grad_norm"] <= 0:
        raise ValueError("max_grad_norm must be greater than 0 when configured")
    if config["log_every_tokens"] <= 0:
        raise ValueError("log_every_tokens must be greater than 0")
    if config["val_every_iterations"] <= 0:
        raise ValueError("val_every_iterations must be greater than 0")
    if config["val_max_iterations"] is not None and config["val_max_iterations"] <= 0:
        raise ValueError("val_max_iterations must be greater than 0 when configured")
    if config["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config["n_epochs"] < 1:
        raise ValueError("n_epochs must be at least 1")
    if config["init_from"] is not None and not config["init_from"].is_file():
        raise ValueError(f"init_from path does not exist: {config['init_from']}")
    if config["resume_from"] is not None and not config["resume_from"].is_file():
        raise ValueError(f"resume_from path does not exist: {config['resume_from']}")
    if config["init_from"] is not None and config["resume_from"] is not None:
        raise ValueError("init_from and resume_from are mutually exclusive")


def get_learning_rate(
    config: dict[str, Any],
    iteration: int,
    max_iterations: int | None = None,
) -> float:
    """Compute the learning rate for a given training iteration."""
    assert iteration > 0, "Iteration must be greater than 0"

    lr_warmup = config["lr_warmup_iterations"]
    learning_rate = config["learning_rate"]
    min_learning_rate = config["min_learning_rate"]
    effective_max = max_iterations if max_iterations is not None else config["max_iterations"]

    if lr_warmup > 0 and iteration <= lr_warmup:
        return learning_rate * iteration / lr_warmup

    if effective_max is None:
        return learning_rate

    decay_iterations = effective_max - lr_warmup
    decay_iteration = max(iteration - lr_warmup - 1, 0)
    decay_progress = min(decay_iteration / decay_iterations, 1.0)
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + cosine_multiplier * (learning_rate - min_learning_rate)


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Update the learning rate in all optimizer parameter groups."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def reset_inference_state(language_model: torch.nn.Module) -> None:
    """Reset the KV-cache and token counter on the transformer decoder."""
    model = getattr(language_model, "_orig_mod", language_model)
    transformer_decoder: TransformerDecoder | None = getattr(model, "transformer_decoder", None)
    assert transformer_decoder is not None, "Invalid LanguageModel instance, transformer_decoder is None."

    transformer_decoder.global_token_counter = 0
    transformer_decoder.kv_cache.clear()

def validate_device(desired_device: str) -> torch.device:
    if desired_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        device = torch.device(desired_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        
        return device


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
    """Save a training checkpoint to disk."""
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
    dataset: TokenizedPreTrainingDataset | TokenizedFinetuneDataset,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_iterations: int | None = None,
) -> float | None:
    """Run a validation pass and return the mean per-token loss."""
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


def train(training_type: str, config: dict[str, Any]) -> None:
    """Run the full pre-training or SFT."""
    validate_config(config)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    if not config["data_dir"].is_dir():
        raise ValueError(f"{config['data_dir']} is not a valid data directory")

    device = validate_device(config["device"])
    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    torch.set_float32_matmul_precision("high")

    tokenizer = TiktokenTokenizer(config["tokenizer_encoding"])

    language_model: LanguageModel = LanguageModel.from_config(
        config["model_config"],
        kv_cache={},
        device=device,
        gradient_checkpointing=config["gradient_checkpointing"],
    ).to(device)

    if language_model.vocab_size < tokenizer.tokenizer.n_vocab:
        raise ValueError(
            "Model's vocab size must be greater than or equal to the tokenizer "
            f"vocabulary size ({tokenizer.tokenizer.n_vocab})"
        )

    if config["compile_model"]:
        language_model = cast(LanguageModel, torch.compile(language_model))

    optimizer = torch.optim.AdamW(
        language_model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)

    optimizer_steps = 0
    train_tokens_seen = 0
    next_log_tokens = config["log_every_tokens"]
    best_val_loss = float("inf")
    wandb_run_id: str | None = None

    if config["init_from"] is not None:
        # Load only model weights from a prior checkpoint (e.g. pre-trained model
        # used as the starting point for SFT). Optimizer state, step counters, and
        # W&B run ID are intentionally NOT restored so this training phase starts clean.
        print(f"Initialising weights from {config['init_from']}...")
        checkpoint = torch.load(config["init_from"], map_location="cpu", weights_only=True)
        model = getattr(language_model, "_orig_mod", language_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint

    if config["resume_from"] is not None:
        checkpoint = torch.load(config["resume_from"], weights_only=True)
        model = getattr(language_model, "_orig_mod", language_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        optimizer_steps = checkpoint["optimizer_steps"]
        train_tokens_seen = checkpoint["train_tokens_seen"]
        best_val_loss = checkpoint["val_loss"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        next_log_tokens = (train_tokens_seen // config["log_every_tokens"] + 1) * config["log_every_tokens"]

    initial_tokens_seen = train_tokens_seen
    train_token_loss = 0.0
    ema_loss: float | None = None
    ema_alpha = 0.98

    if wandb_run_id is None:
        wandb_run_id = config["wandb_resume_id"]

    wandb_current_run = None
    if config["wandb_enabled"]:
        wandb_current_run = wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            mode=config["wandb_mode"],  # type: ignore[arg-type]
            id=config["wandb_resume_id"] or wandb_run_id,
            resume="must" if wandb_run_id is not None else None,
            config={
                "training": training_config_to_dict(config),
                "model": model_config_to_dict(language_model),
            },
        )

    skip_until_step: int = 0
    if config["wandb_enabled"] and wandb_run_id is not None and wandb_current_run is not None:
        skip_until_step = int(wandb_current_run.summary.get("train/tokens_seen", 0))

    try:
        if training_type == "sft":
            dataset: TokenizedPreTrainingDataset | TokenizedFinetuneDataset = TokenizedFinetuneDataset(
                data_dir=config["data_dir"],
                sequence_length=language_model.sequence_length,
                batch_size=config["batch_size"],
                pad_token_id=tokenizer.get_end_token(),
            )
            assert isinstance(dataset, TokenizedFinetuneDataset)
            n_optimizer_steps_per_epoch = max(
                dataset.n_samples["train"] // (config["batch_size"] * config["gradient_accumulation_steps"]),
                1,
            )
            computed_max = config["n_epochs"] * n_optimizer_steps_per_epoch
            effective_max_iterations: int | None = (
                min(config["max_iterations"], computed_max)
                if config["max_iterations"] is not None
                else computed_max
            )
        else:
            dataset = TokenizedPreTrainingDataset(
                data_dir=config["data_dir"],
                sequence_length=language_model.sequence_length,
                batch_size=config["batch_size"],
            )

            if config["resume_from"] is not None:
                dataset._offset["train"] = train_tokens_seen

            if config["max_iterations"] is not None:
                effective_max_iterations = config["max_iterations"]
            elif config["n_epochs"] > 1:
                n_train_tokens = len(dataset._mmap["train"])
                tokens_per_step = (
                    config["batch_size"]
                    * language_model.sequence_length
                    * config["gradient_accumulation_steps"]
                )
                n_optimizer_steps_per_epoch = max(n_train_tokens // tokens_per_step, 1)
                effective_max_iterations = config["n_epochs"] * n_optimizer_steps_per_epoch
            else:
                effective_max_iterations = None

        if config["resume_from"] is not None:
            set_optimizer_learning_rate(
                optimizer,
                get_learning_rate(config, optimizer_steps, effective_max_iterations),
            )

        progress = tqdm(
            desc="\033[1mTraining\033[0m",
            unit=" total tokens",
            bar_format="{desc}: {n_fmt}{unit} [elapsed: {elapsed}, {rate_fmt}{postfix}]",
            initial=train_tokens_seen,
        )

        for epoch in range(1, config["n_epochs"] + 1):
            skip_reset = (
                config["resume_from"] is not None
                and epoch == 1
                and training_type == "pretraining"
            )
            if not skip_reset:
                dataset.reset_epoch(seed=config["seed"] + epoch)

            while True:
                if config["max_iterations"] is not None and optimizer_steps >= config["max_iterations"]:
                    break

                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                tokens_in_optimizer_step = 0
                epoch_ended = False

                for _ in range(config["gradient_accumulation_steps"]):
                    try:
                        inputs, targets = dataset.get_sequential_batch("train")
                    except StopIteration:
                        epoch_ended = True
                        break
                    tokens_in_micro_batch = inputs.numel()
                    language_model.train()
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        loss = language_model(inputs, targets=targets)

                    loss_scale = 1 / config["gradient_accumulation_steps"]
                    scaler.scale(loss * loss_scale).backward()

                    accumulated_loss += float(loss.detach().item())
                    tokens_in_optimizer_step += tokens_in_micro_batch

                if tokens_in_optimizer_step > 0:
                    optimizer_steps += 1
                    learning_rate = get_learning_rate(config, optimizer_steps, effective_max_iterations)
                    set_optimizer_learning_rate(optimizer, learning_rate)
                    if config["max_grad_norm"] is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()

                    loss = accumulated_loss / config["gradient_accumulation_steps"]
                    train_tokens_seen += tokens_in_optimizer_step
                    train_token_loss += loss * tokens_in_optimizer_step
                    avg_train_loss = train_token_loss / (train_tokens_seen - initial_tokens_seen)
                    ema_loss = (
                        loss if ema_loss is None
                        else ema_alpha * ema_loss + (1 - ema_alpha) * loss
                    )
                    progress.update(tokens_in_optimizer_step)
                    progress.set_postfix(loss=f"{ema_loss:.4f}", lr=f"{learning_rate:.2e}")

                    if train_tokens_seen >= next_log_tokens and train_tokens_seen > skip_until_step:
                        if config["wandb_enabled"] and wandb_current_run is not None:
                            wandb_current_run.log(
                                data={
                                    "train/loss": loss,
                                    "train/loss_ema": ema_loss,
                                    "train/avg_loss": avg_train_loss,
                                    "train/tokens_seen": train_tokens_seen,
                                    "train/optimizer_steps": optimizer_steps,
                                    "train/learning_rate": learning_rate,
                                },
                                step=train_tokens_seen,
                            )
                        while next_log_tokens <= train_tokens_seen:
                            next_log_tokens += config["log_every_tokens"]

                    if optimizer_steps % config["val_every_iterations"] == 0:
                        step_val_loss = validate(
                            language_model=language_model,
                            dataset=dataset,
                            device=device,
                            amp_dtype=amp_dtype,
                            use_amp=use_amp,
                            max_iterations=config["val_max_iterations"],
                        )
                        language_model.train()
                        if step_val_loss is not None:
                            if config["wandb_enabled"] and wandb_current_run is not None and train_tokens_seen > skip_until_step:
                                wandb_current_run.log(
                                    data={
                                        "val/loss": step_val_loss,
                                        "train/tokens_seen": train_tokens_seen,
                                        "train/optimizer_steps": optimizer_steps,
                                    },
                                    step=train_tokens_seen,
                                )
                            save_checkpoint(
                                path=config["checkpoint_dir"] / "checkpoint_val_last.pt",
                                language_model=language_model,
                                optimizer=optimizer,
                                scaler=scaler,
                                optimizer_steps=optimizer_steps,
                                train_tokens_seen=train_tokens_seen,
                                train_token_loss=train_token_loss,
                                val_loss=step_val_loss,
                                wandb_run_id=wandb_current_run.id if wandb_current_run is not None else None,
                            )
                            if step_val_loss < best_val_loss:
                                best_val_loss = step_val_loss
                                save_checkpoint(
                                    path=config["checkpoint_dir"] / "checkpoint_best.pt",
                                    language_model=language_model,
                                    optimizer=optimizer,
                                    scaler=scaler,
                                    optimizer_steps=optimizer_steps,
                                    train_tokens_seen=train_tokens_seen,
                                    train_token_loss=train_token_loss,
                                    val_loss=step_val_loss,
                                    wandb_run_id=wandb_current_run.id if wandb_current_run is not None else None,
                                )

                if epoch_ended:
                    if config["wandb_enabled"] and wandb_current_run is not None and config["n_epochs"] > 1 and train_tokens_seen > skip_until_step:
                        wandb_current_run.log(
                            data={
                                "train/epoch": epoch,
                                "train/tokens_seen": train_tokens_seen,
                            },
                            step=train_tokens_seen,
                        )
                    break

        progress.close()

        if config["wandb_enabled"] and wandb_current_run is not None and train_tokens_seen > 0 and train_tokens_seen > skip_until_step:
            wandb_current_run.log(
                data={
                    "train/final_loss": train_token_loss / train_tokens_seen,
                    "train/tokens_seen": train_tokens_seen,
                    "train/optimizer_steps": optimizer_steps,
                    "train/learning_rate": get_learning_rate(config, optimizer_steps, effective_max_iterations),
                },
                step=train_tokens_seen,
            )

        final_val_loss = validate(
            language_model=language_model,
            dataset=dataset,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            max_iterations=config["val_max_iterations"],
        )
        language_model.train()
        if config["wandb_enabled"] and wandb_current_run is not None and final_val_loss is not None and max(train_tokens_seen, 1) > skip_until_step:
            wandb_current_run.log(
                data={
                    "val/loss": final_val_loss,
                    "train/tokens_seen": train_tokens_seen,
                    "train/optimizer_steps": optimizer_steps,
                },
                step=max(train_tokens_seen, 1),
            )

        save_checkpoint(
            path=config["checkpoint_dir"] / "checkpoint_final.pt",
            language_model=language_model,
            optimizer=optimizer,
            scaler=scaler,
            optimizer_steps=optimizer_steps,
            train_tokens_seen=train_tokens_seen,
            train_token_loss=train_token_loss,
            val_loss=final_val_loss if final_val_loss is not None else float("nan"),
            wandb_run_id=wandb_current_run.id if wandb_current_run is not None else None,
        )
    finally:
        if wandb_current_run is not None:
            wandb_current_run.finish()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training entry point."""
    parser = argparse.ArgumentParser(description="Train the piccolo language model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--training-type",
        type=str,
        choices=["pretraining", "sft"],
        default="pretraining",
        help="Type of the training to run (Pre-training or SFT)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_training_config(args.config)
    train(args.training_type, config)
