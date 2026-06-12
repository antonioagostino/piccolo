import argparse
import math
import random
from pathlib import Path
from typing import Any, Union, cast

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

DATA_DIR: Path
MODEL_CONFIG: Path
TOKENIZER_ENCODING: str
BATCH_SIZE: int
LEARNING_RATE: float
MIN_LEARNING_RATE: float
LR_WARMUP_ITERATIONS: int
MAX_ITERATIONS: int | None
WEIGHT_DECAY: float
MAX_GRAD_NORM: float | None
DEVICE: str
COMPILE_MODEL: bool
GRADIENT_CHECKPOINTING: bool
GRADIENT_ACCUMULATION_STEPS: int
SEED: int
LOG_EVERY_TOKENS: int
VAL_EVERY_ITERATIONS: int
VAL_MAX_ITERATIONS: int | None
N_EPOCHS: int
CHECKPOINT_DIR: Path
INIT_FROM: Path | None
RESUME_FROM: Path | None
WANDB_ENABLED: bool
WANDB_PROJECT: str | None
WANDB_RUN_NAME: str | None
WANDB_MODE: str | None
WANDB_RESUME_ID: str | None


def training_config_to_dict() -> dict[str, Any]:
    """Serialise global training config variables to a plain dict."""
    return {
        "data_dir": str(DATA_DIR),
        "model_config": str(MODEL_CONFIG),
        "tokenizer_encoding": TOKENIZER_ENCODING,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "min_learning_rate": MIN_LEARNING_RATE,
        "lr_warmup_iterations": LR_WARMUP_ITERATIONS,
        "max_iterations": MAX_ITERATIONS,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "device": DEVICE,
        "compile_model": COMPILE_MODEL,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "seed": SEED,
        "log_every_tokens": LOG_EVERY_TOKENS,
        "val_every_iterations": VAL_EVERY_ITERATIONS,
        "val_max_iterations": VAL_MAX_ITERATIONS,
        "n_epochs": N_EPOCHS,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "init_from": str(INIT_FROM) if INIT_FROM is not None else None,
        "resume_from": str(RESUME_FROM) if RESUME_FROM is not None else None,
        "wandb": {
            "enabled": WANDB_ENABLED,
            "project": WANDB_PROJECT,
            "run_name": WANDB_RUN_NAME,
            "mode": WANDB_MODE,
        },
    }


def model_config_to_dict(language_model: LanguageModel) -> dict[str, int | float | None]:
    """Serialize input LanguageModel config to a dictionary"""
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


def load_training_config(config_path: Path) -> None:
    """Parse config YAML file and create global training config variables"""
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    assert isinstance(raw_config, dict), "Invalid YAML mapping"
    training_config = raw_config["training"]
    assert isinstance(training_config, dict), "Invalid YAML mapping"
    wandb_config = training_config["wandb"]
    assert isinstance(wandb_config, dict), "Invalid YAML mapping"

    global DATA_DIR, MODEL_CONFIG, TOKENIZER_ENCODING, BATCH_SIZE, LEARNING_RATE
    global MIN_LEARNING_RATE, LR_WARMUP_ITERATIONS, MAX_ITERATIONS, WEIGHT_DECAY
    global MAX_GRAD_NORM, DEVICE, COMPILE_MODEL, GRADIENT_CHECKPOINTING
    global GRADIENT_ACCUMULATION_STEPS, SEED, LOG_EVERY_TOKENS, VAL_EVERY_ITERATIONS
    global VAL_MAX_ITERATIONS, N_EPOCHS, CHECKPOINT_DIR, INIT_FROM, RESUME_FROM
    global WANDB_ENABLED, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_MODE, WANDB_RESUME_ID

    if (_v := training_config.get("data_dir", None)) is None:
        raise ValueError("Missing 'data_dir' in training config file")
    DATA_DIR = Path(_v)

    if (_v := training_config.get("model_config", None)) is None:
        raise ValueError("Missing 'model_config' in training config file")
    MODEL_CONFIG = Path(_v)

    if (_v := training_config.get("tokenizer_encoding", None)) is None:
        raise ValueError("Missing 'tokenizer_encoding' in training config file")
    TOKENIZER_ENCODING = str(_v)

    if (_v := training_config.get("batch_size", None)) is None:
        raise ValueError("Missing 'batch_size' in training config file")
    BATCH_SIZE = int(_v)

    if (_v := training_config.get("learning_rate", None)) is None:
        raise ValueError("Missing 'learning_rate' in training config file")
    LEARNING_RATE = float(_v)

    if (_v := training_config.get("min_learning_rate", None)) is None:
        raise ValueError("Missing 'min_learning_rate' in training config file")
    MIN_LEARNING_RATE = float(_v)

    if (_v := training_config.get("lr_warmup_iterations", None)) is None:
        raise ValueError("Missing 'lr_warmup_iterations' in training config file")
    LR_WARMUP_ITERATIONS = int(_v)

    _v = training_config.get("max_iterations", None)
    MAX_ITERATIONS = None if _v is None else int(_v)

    if (_v := training_config.get("weight_decay", None)) is None:
        raise ValueError("Missing 'weight_decay' in training config file")
    WEIGHT_DECAY = float(_v)

    MAX_GRAD_NORM = training_config.get("max_grad_norm", None)

    if (_v := training_config.get("device", None)) is None:
        raise ValueError("Missing 'device' in training config file")
    DEVICE = str(_v)

    if (_v := training_config.get("compile_model", None)) is None:
        raise ValueError("Missing 'compile_model' in training config file")
    COMPILE_MODEL = bool(_v)

    if (_v := training_config.get("gradient_checkpointing", None)) is None:
        raise ValueError("Missing 'gradient_checkpointing' in training config file")
    GRADIENT_CHECKPOINTING = bool(_v)

    if (_v := training_config.get("gradient_accumulation_steps", None)) is None:
        raise ValueError("Missing 'gradient_accumulation_steps' in training config file")
    GRADIENT_ACCUMULATION_STEPS = int(_v)

    if (_v := training_config.get("seed", None)) is None:
        raise ValueError("Missing 'seed' in training config file")
    SEED = int(_v)

    if (_v := training_config.get("log_every_tokens", None)) is None:
        raise ValueError("Missing 'log_every_tokens' in training config file")
    LOG_EVERY_TOKENS = int(_v)

    if (_v := training_config.get("val_every_iterations", None)) is None:
        raise ValueError("Missing 'val_every_iterations' in training config file")
    VAL_EVERY_ITERATIONS = int(_v)

    _v = training_config.get("val_max_iterations", None)
    VAL_MAX_ITERATIONS = None if _v is None else int(_v)

    if (_v := training_config.get("n_epochs", None)) is None:
        raise ValueError("Missing 'n_epochs' in training config file")
    N_EPOCHS = int(_v)

    if (_v := training_config.get("checkpoint_dir", None)) is None:
        raise ValueError("Missing 'checkpoint_dir' in training config file")
    CHECKPOINT_DIR = Path(_v)

    _v = training_config.get("init_from", None)
    INIT_FROM = Path(_v) if _v is not None else None

    _v = training_config.get("resume_from", None)
    RESUME_FROM = Path(_v) if _v is not None else None

    if (_v := wandb_config.get("enabled", None)) is None:
        raise ValueError("Missing 'enabled' in wandb config")
    WANDB_ENABLED = bool(_v)

    if WANDB_ENABLED:
        if (_v := wandb_config.get("project", None)) is None:
            raise ValueError("Missing 'project' in wandb config")
        WANDB_PROJECT = str(_v)
        WANDB_RUN_NAME = wandb_config.get("run_name", None)
        if (_v := wandb_config.get("mode", None)) is None:
            raise ValueError("Missing 'mode' in wandb config")
        WANDB_MODE = str(_v)
        WANDB_RESUME_ID = wandb_config.get("resume_id", None)
    else:
        WANDB_PROJECT = None
        WANDB_RUN_NAME = None
        WANDB_MODE = None
        WANDB_RESUME_ID = None

def validate_config_vars() -> None:
    """Check global training variables"""
    if BATCH_SIZE <= 0:
        raise ValueError("batch_size must be greater than 0")
    if LEARNING_RATE <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if MIN_LEARNING_RATE < 0:
        raise ValueError("min_learning_rate must be greater than or equal to 0")
    if MIN_LEARNING_RATE > LEARNING_RATE:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")
    if LR_WARMUP_ITERATIONS < 0:
        raise ValueError("lr_warmup_iterations must be greater than or equal to 0")
    if MAX_ITERATIONS is not None and MAX_ITERATIONS <= 0:
        raise ValueError("max_iterations must be greater than 0 when configured")
    if MAX_ITERATIONS is not None and MAX_ITERATIONS <= LR_WARMUP_ITERATIONS:
        raise ValueError("max_iterations must be greater than lr_warmup_iterations")
    if WEIGHT_DECAY < 0:
        raise ValueError("weight_decay must be greater than or equal to 0")
    if MAX_GRAD_NORM is not None and MAX_GRAD_NORM <= 0:
        raise ValueError("max_grad_norm must be greater than 0 when configured")
    if LOG_EVERY_TOKENS <= 0:
        raise ValueError("log_every_tokens must be greater than 0")
    if VAL_EVERY_ITERATIONS <= 0:
        raise ValueError("val_every_iterations must be greater than 0")
    if VAL_MAX_ITERATIONS is not None and VAL_MAX_ITERATIONS <= 0:
        raise ValueError("val_max_iterations must be greater than 0 when configured")
    if GRADIENT_ACCUMULATION_STEPS < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if N_EPOCHS < 1:
        raise ValueError("n_epochs must be at least 1")
    if INIT_FROM is not None and not INIT_FROM.is_file():
        raise ValueError(f"init_from path does not exist: {INIT_FROM}")
    if RESUME_FROM is not None and not RESUME_FROM.is_file():
        raise ValueError(f"resume_from path does not exist: {RESUME_FROM}")
    if INIT_FROM is not None and RESUME_FROM is not None:
        raise ValueError("init_from and resume_from are mutually exclusive")
    


def get_learning_rate(
    iteration: int,
    max_iterations: int | None = None,
) -> float:
    """Compute the learning rate for a given training iteration."""
    assert iteration > 0, "Iteration must be greater than 0"

    if LR_WARMUP_ITERATIONS > 0 and iteration <= LR_WARMUP_ITERATIONS:
        return LEARNING_RATE * iteration / LR_WARMUP_ITERATIONS

    effective_max = max_iterations if max_iterations is not None else MAX_ITERATIONS
    if effective_max is None:
        return LEARNING_RATE

    decay_iterations = effective_max - LR_WARMUP_ITERATIONS
    decay_iteration = max(iteration - LR_WARMUP_ITERATIONS - 1, 0)
    decay_progress = min(decay_iteration / decay_iterations, 1.0)
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return MIN_LEARNING_RATE + cosine_multiplier * (
        LEARNING_RATE - MIN_LEARNING_RATE
    )


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


def train(training_type: str) -> None:
    """Run the full pre-training or SFT"""
    validate_config_vars()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    
    if not DATA_DIR.is_dir():
        raise ValueError(f"{DATA_DIR} is not a valid data directory")

    if DEVICE == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(DEVICE)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")

    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    torch.set_float32_matmul_precision("high")

    tokenizer = TiktokenTokenizer(TOKENIZER_ENCODING)

    language_model: LanguageModel = LanguageModel.from_config(
        MODEL_CONFIG,
        kv_cache={},
        device=device,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    ).to(device)

    if language_model.vocab_size < tokenizer.tokenizer.n_vocab:
        raise ValueError(
            "Model's vocab size must be greater than or equal to the tokenizer "
            f"vocabulary size ({tokenizer.tokenizer.n_vocab})"
        )
    
    if COMPILE_MODEL:
        language_model = cast(LanguageModel, torch.compile(language_model))

    optimizer = torch.optim.AdamW(
        language_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)

    optimizer_steps = 0
    train_tokens_seen = 0
    next_log_tokens = LOG_EVERY_TOKENS
    best_val_loss = float("inf")
    wandb_run_id: str | None = None

    if INIT_FROM is not None:
        # Load only the model weights from a prior checkpoint (e.g. a
        # pre-trained model used as the starting point for SFT).
        # Optimizer state, step counters, and W&B run ID are intentionally
        # NOT restored so the new training phase starts from a clean slate.
        print(f"Initialising weights from {INIT_FROM}...")
        checkpoint = torch.load(INIT_FROM, map_location="cpu", weights_only=True)
        model = getattr(language_model, "_orig_mod", language_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint

    if RESUME_FROM is not None:
        checkpoint = torch.load(RESUME_FROM, weights_only=True)
        model = getattr(language_model, "_orig_mod", language_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        optimizer_steps = checkpoint["optimizer_steps"]
        train_tokens_seen = checkpoint["train_tokens_seen"]
        best_val_loss = checkpoint["val_loss"]
        wandb_run_id = checkpoint.get("wandb_run_id")
        next_log_tokens = (train_tokens_seen // LOG_EVERY_TOKENS + 1) * LOG_EVERY_TOKENS
        # LR is restored after effective_max_iterations is computed below.

    # Track loss/tokens from this run's start so avg_loss is always meaningful,
    # even when resuming from a checkpoint with a large initial train_tokens_seen.
    initial_tokens_seen = train_tokens_seen
    train_token_loss = 0.0
    ema_loss: float | None = None
    ema_alpha = 0.98

    if wandb_run_id is None:
        wandb_run_id = WANDB_RESUME_ID

    wandb_current_run = None
    if WANDB_ENABLED:
        wandb_current_run = wandb.init(project=WANDB_PROJECT,
                                    name=WANDB_RUN_NAME,
                                    mode=WANDB_MODE,  # type: ignore[arg-type]
                                    id=WANDB_RESUME_ID or wandb_run_id,
                                    resume="must" if wandb_run_id is not None else None,
                                    config={
                                        "training": training_config_to_dict(),
                                        "model": model_config_to_dict(language_model)
                                        }
        )

    # When resuming, skip any step already present in the run so we never
    # log backwards and trigger a step-ordering error from W&B.
    skip_until_step: int = 0
    if WANDB_ENABLED and wandb_run_id is not None and wandb_current_run is not None:
        skip_until_step = int(wandb_current_run.summary.get("train/tokens_seen", 0))

    try:
        if training_type == "sft":
            dataset: TokenizedPreTrainingDataset | TokenizedFinetuneDataset = TokenizedFinetuneDataset(
                data_dir=DATA_DIR,
                sequence_length=language_model.sequence_length,
                batch_size=BATCH_SIZE,
                pad_token_id=tokenizer.get_end_token(),
                seed=SEED,
            )
            assert isinstance(dataset, TokenizedFinetuneDataset)
            n_optimizer_steps_per_epoch = max(
                dataset.n_samples["train"] // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
                1,
            )
            computed_max = N_EPOCHS * n_optimizer_steps_per_epoch
            effective_max_iterations: int | None = (
                min(MAX_ITERATIONS, computed_max)
                if MAX_ITERATIONS is not None
                else computed_max
            )
        else:
            dataset = TokenizedPreTrainingDataset(
                data_dir=DATA_DIR,
                sequence_length=language_model.sequence_length,
                batch_size=BATCH_SIZE,
            )

            if RESUME_FROM is not None:
                # Restore position within the current epoch for seamless resume.
                dataset._offset["train"] = train_tokens_seen
            
            if MAX_ITERATIONS is not None:
                effective_max_iterations = MAX_ITERATIONS
            elif N_EPOCHS > 1:
                n_train_tokens = len(dataset._mmap["train"])
                tokens_per_step = (
                    BATCH_SIZE
                    * language_model.sequence_length
                    * GRADIENT_ACCUMULATION_STEPS
                )
                n_optimizer_steps_per_epoch = max(n_train_tokens // tokens_per_step, 1)
                effective_max_iterations = N_EPOCHS * n_optimizer_steps_per_epoch
            else:
                effective_max_iterations = None

        # Now that effective_max_iterations is known, restore LR for resumed runs.
        if RESUME_FROM is not None:
            set_optimizer_learning_rate(
                optimizer,
                get_learning_rate(optimizer_steps, effective_max_iterations),
            )

        progress = tqdm(
            desc="\033[1mTraining\033[0m",
            unit=" total tokens",
            bar_format="{desc}: {n_fmt}{unit} [elapsed: {elapsed}, {rate_fmt}{postfix}]",
            initial=train_tokens_seen,
        )

        for epoch in range(1, N_EPOCHS + 1):
            # On a resumed pre-training run the data offset is already set
            # above; skip the reset so we continue from where we left off.
            # For all other cases (epoch > 1, or finetune), do the reset.
            skip_reset = (
                RESUME_FROM is not None
                and epoch == 1
                and training_type == "pretraining"
            )
            if not skip_reset:
                # Use a per-epoch seed derived from the base seed so each epoch
                # sees a different sample order.
                dataset.reset_epoch(seed=SEED + epoch)

            while True:
                # Honour an explicit max_iterations hard cap if set.
                # For epoch-based training the for loop above is the natural
                # stopping mechanism; effective_max_iterations is used only
                # for LR scheduling and must not be the stopping criterion
                # (it is a floor-rounded estimate and can be off by n_epochs).
                if MAX_ITERATIONS is not None and optimizer_steps >= MAX_ITERATIONS:
                    break

                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                tokens_in_optimizer_step = 0
                epoch_ended = False

                for _ in range(GRADIENT_ACCUMULATION_STEPS):
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

                    # Scaling is only needed for CUDA float16. With bfloat16 or CPU,
                    # GradScaler is disabled and these calls are no-ops/pass-throughs.
                    
                    # Multiplies the loss by loss_scale before backpropagation; set
                    # loss_scale = 1 / gradient_accumulation_steps so gradients average
                    # correctly across an accumulation window.
                    loss_scale = 1 / GRADIENT_ACCUMULATION_STEPS
                    scaler.scale(loss * loss_scale).backward()

                    accumulated_loss += float(loss.detach().item())
                    tokens_in_optimizer_step += tokens_in_micro_batch

                # Flush any partial accumulation collected before epoch end.
                if tokens_in_optimizer_step > 0:
                    optimizer_steps += 1
                    learning_rate = get_learning_rate(
                        optimizer_steps, effective_max_iterations
                    )
                    set_optimizer_learning_rate(optimizer, learning_rate)
                    if MAX_GRAD_NORM is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()

                    loss = accumulated_loss / GRADIENT_ACCUMULATION_STEPS
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
                        if WANDB_ENABLED and wandb_current_run is not None:
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
                            next_log_tokens += LOG_EVERY_TOKENS

                    if optimizer_steps % VAL_EVERY_ITERATIONS == 0:
                        step_val_loss = validate(
                            language_model=language_model,
                            dataset=dataset,
                            device=device,
                            amp_dtype=amp_dtype,
                            use_amp=use_amp,
                            max_iterations=VAL_MAX_ITERATIONS,
                        )
                        language_model.train()
                        if step_val_loss is not None:
                            if WANDB_ENABLED and wandb_current_run is not None and train_tokens_seen > skip_until_step:
                                wandb_current_run.log(
                                    data={
                                        "val/loss": step_val_loss,
                                        "train/tokens_seen": train_tokens_seen,
                                        "train/optimizer_steps": optimizer_steps,
                                    },
                                    step=train_tokens_seen,
                                )
                            save_checkpoint(
                                path=CHECKPOINT_DIR / "checkpoint_val_last.pt",
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
                                    path=CHECKPOINT_DIR / "checkpoint_best.pt",
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
                    # Log epoch completion and break out of the while loop so
                    # the outer for loop can advance to the next epoch.
                    if WANDB_ENABLED and wandb_current_run is not None and N_EPOCHS > 1 and train_tokens_seen > skip_until_step:
                        wandb_current_run.log(
                            data={
                                "train/epoch": epoch,
                                "train/tokens_seen": train_tokens_seen,
                            },
                            step=train_tokens_seen,
                        )
                    break

        progress.close()

        if WANDB_ENABLED and wandb_current_run is not None and train_tokens_seen > 0 and train_tokens_seen > skip_until_step:
            wandb_current_run.log(
                data={
                    "train/final_loss": train_token_loss / train_tokens_seen,
                    "train/tokens_seen": train_tokens_seen,
                    "train/optimizer_steps": optimizer_steps,
                    "train/learning_rate": get_learning_rate(optimizer_steps, effective_max_iterations),
                },
                step=train_tokens_seen,
            )

        final_val_loss = validate(
            language_model=language_model,
            dataset=dataset,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            max_iterations=VAL_MAX_ITERATIONS,
        )
        language_model.train()
        if WANDB_ENABLED and wandb_current_run is not None and final_val_loss is not None and max(train_tokens_seen, 1) > skip_until_step:
            wandb_current_run.log(
                data={
                    "val/loss": final_val_loss,
                    "train/tokens_seen": train_tokens_seen,
                    "train/optimizer_steps": optimizer_steps,
                },
                step=max(train_tokens_seen, 1),
            )

        save_checkpoint(
            path=CHECKPOINT_DIR / "checkpoint_final.pt",
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
        default="pratraining",
        help="Type of the training to run (Pre-training or SFT)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_training_config(args.config)
    train(args.training_type)
