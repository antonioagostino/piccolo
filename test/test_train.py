from pathlib import Path

import pandas as pd
import pytest

from src.train import get_learning_rate, load_training_config, train
from src.training_budget import estimate_token_budget


def test_train_runs_from_yaml_config(tmp_path: Path):
    data_dir = tmp_path / "data"
    dataset_file = data_dir / "raw_text" / "CulturaX" / "dataset_file.parquet"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": [
                "hello world " * 20,
                "small language model training sample " * 20,
            ]
        }
    ).to_parquet(dataset_file, engine="auto", index=False)

    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 100277",
                "  sequence_length: 4",
                "  embedding_dim: 16",
                "  n_decoder_blocks: 1",
                "  n_heads: 4",
                "  n_kv_heads: 2",
                "  ffn_hidden_dim: 32",
                "  dropout_rate: 0.0",
            ]
        ),
        encoding="utf-8",
    )

    training_config_path = tmp_path / "training.yaml"
    training_config_path.write_text(
        "\n".join(
            [
                "training:",
                f"  data_dir: {data_dir}",
                f"  model_config: {model_config_path}",
                "  train_split_size: 0.5",
                "  batch_size: 1",
                "  learning_rate: 0.001",
                "  min_learning_rate: 0.0001",
                "  lr_warmup_iterations: 2",
                "  max_iterations: 6",
                "  weight_decay: 0.0",
                "  max_grad_norm: 1.0",
                "  device: cpu",
                "  compile_model: false",
                "  seed: 0",
                "  log_every_tokens: 1",
                "  wandb:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    train(load_training_config(training_config_path))


def test_learning_rate_uses_warmup_then_cosine_decay(tmp_path: Path):
    training_config_path = tmp_path / "training.yaml"
    training_config_path.write_text(
        "\n".join(
            [
                "training:",
                "  learning_rate: 1.0",
                "  min_learning_rate: 0.1",
                "  lr_warmup_iterations: 2",
                "  max_iterations: 6",
                "  wandb:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_training_config(training_config_path)

    assert get_learning_rate(config, 1) == pytest.approx(0.5)
    assert get_learning_rate(config, 2) == pytest.approx(1.0)
    assert get_learning_rate(config, 3) == pytest.approx(1.0)
    assert get_learning_rate(config, 5) == pytest.approx(0.55)
    assert get_learning_rate(config, 7) == pytest.approx(0.1)


def test_estimate_token_budget_returns_full_pass_iterations(tmp_path: Path):
    data_dir = tmp_path / "data"
    dataset_file = data_dir / "raw_text" / "CulturaX" / "dataset_file.parquet"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": ["hello world " * 10, "training tokens " * 10]}).to_parquet(
        dataset_file,
        engine="auto",
        index=False,
    )
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 100277",
                "  sequence_length: 4",
                "  embedding_dim: 16",
                "  n_decoder_blocks: 1",
                "  n_heads: 4",
                "  n_kv_heads: 2",
                "  ffn_hidden_dim: 32",
                "  dropout_rate: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    training_config_path = tmp_path / "training.yaml"
    training_config_path.write_text(
        "\n".join(
            [
                "training:",
                f"  data_dir: {data_dir}",
                f"  model_config: {model_config_path}",
                "  train_split_size: 1.0",
                "  batch_size: 2",
                "  wandb:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    estimate = estimate_token_budget(load_training_config(training_config_path))

    assert estimate.tokens_per_iteration == 8
    assert estimate.train_iterations == estimate.split_token_counts.train_tokens // 8
