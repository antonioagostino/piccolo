from __future__ import annotations

from dataclasses import dataclass

from src.dataset import SplitTokenCounts, count_pre_training_tokens
from src.tokenizer import TiktokenTokenizer
from src.train import TrainingConfig, resolve_data_dir, validate_config
from src.transformer import ModelConfig


@dataclass(frozen=True)
class TokenBudgetEstimate:
    split_token_counts: SplitTokenCounts
    tokens_per_iteration: int
    train_iterations: int
    val_iterations: int


def estimate_token_budget(config: TrainingConfig) -> TokenBudgetEstimate:
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
    return TokenBudgetEstimate(
        split_token_counts=split_token_counts,
        tokens_per_iteration=tokens_per_iteration,
        train_iterations=split_token_counts.train_tokens // tokens_per_iteration,
        val_iterations=split_token_counts.val_tokens // tokens_per_iteration,
    )


def format_token_budget_estimate(estimate: TokenBudgetEstimate) -> str:
    return "\n".join(
        [
            f"train_tokens: {estimate.split_token_counts.train_tokens}",
            f"val_tokens: {estimate.split_token_counts.val_tokens}",
            f"tokens_per_iteration: {estimate.tokens_per_iteration}",
            f"suggested max_iterations: {estimate.train_iterations}",
            f"validation_iterations: {estimate.val_iterations}",
        ]
    )


def print_token_budget_estimate(config: TrainingConfig) -> None:
    print(format_token_budget_estimate(estimate_token_budget(config)))
