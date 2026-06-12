from pathlib import Path

import pandas as pd
import pytest

from src.train import get_learning_rate, load_training_config, train
from src.utils.tokenize_pretraining_dataset import tokenize_dataset


def test_train_runs_from_yaml_config(tmp_path: Path):
    raw_dir = tmp_path / "data" / "raw_text"
    dataset_file = raw_dir / "CulturaX" / "dataset_file.parquet"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": [
                "hello world " * 20,
                "small language model training sample " * 20,
            ]
        }
    ).to_parquet(dataset_file, engine="auto", index=False)

    tokenized_dir = tmp_path / "data" / "raw_text_tokenized"
    tokenize_dataset(
        data_dir=raw_dir,
        output_dir=tokenized_dir,
        encoding="cl100k_base",
        train_split=0.5,
        seed=0,
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
    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    training_config_path.write_text(
        "\n".join(
            [
                "training:",
                f"  data_dir: {tokenized_dir}",
                f"  model_config: {model_config_path}",
                "  tokenizer_encoding: cl100k_base",
                "  gradient_checkpointing: false",
                "  gradient_accumulation_steps: 1",
                "  batch_size: 1",
                "  learning_rate: 0.001",
                "  min_learning_rate: 0.0001",
                "  lr_warmup_iterations: 2",
                "  max_iterations: 6",
                "  weight_decay: 0.0",
                "  max_grad_norm: 1.0",
                "  device: cpu",
                "  compile_model: false",
                "  val_every_iterations: 10",
                "  seed: 0",
                "  val_max_iterations: 5",
                "  n_epochs: 1",
                f"  checkpoint_dir: {checkpoint_path}",
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
                f"  data_dir: {training_config_path}",
                f"  model_config: {training_config_path}",
                "  tokenizer_encoding: cl100k_base",
                "  gradient_checkpointing: false",
                "  gradient_accumulation_steps: 1",
                "  batch_size: 1",
                "  learning_rate: 1.0",
                "  min_learning_rate: 0.1",
                "  lr_warmup_iterations: 2",
                "  max_iterations: 6",
                "  weight_decay: 0.0",
                "  max_grad_norm: 1.0",
                "  device: cpu",
                "  compile_model: false",
                "  val_every_iterations: 10",
                "  seed: 0",
                "  val_max_iterations: 5",
                "  n_epochs: 1",
                f"  checkpoint_dir: {training_config_path}",
                "  log_every_tokens: 1",
                "  wandb:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_training_config(training_config_path)

    assert get_learning_rate(1) == pytest.approx(0.5)
    assert get_learning_rate(2) == pytest.approx(1.0)
    assert get_learning_rate(3) == pytest.approx(1.0)
    assert get_learning_rate(5) == pytest.approx(0.55)
    assert get_learning_rate(7) == pytest.approx(0.1)


def test_tokenized_dataset_token_counts_match_metadata(tmp_path: Path):
    raw_dir = tmp_path / "raw_text"
    dataset_file = raw_dir / "CulturaX" / "dataset_file.parquet"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": ["hello world " * 10, "training tokens " * 10]}).to_parquet(
        dataset_file,
        engine="auto",
        index=False,
    )

    tokenized_dir = tmp_path / "raw_text_tokenized"
    tokenize_dataset(
        data_dir=raw_dir,
        output_dir=tokenized_dir,
        encoding="cl100k_base",
        train_split=1.0,
        seed=0,
    )

    import json, numpy as np
    meta = json.loads((tokenized_dir / "metadata.json").read_text())
    train_tokens = np.memmap(tokenized_dir / "train.bin", dtype="uint32", mode="r")

    assert meta["train_tokens"] == len(train_tokens)
    assert meta["val_tokens"] == 0
