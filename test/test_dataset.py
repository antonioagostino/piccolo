import pytest
import pandas as pd
from pathlib import Path
import torch
from src.dataset import PreTrainingDataset, count_pre_training_tokens, load_text_from_parquet_file
from src.tokenizer import TiktokenTokenizer, Tokenizer


class LengthTokenizer(Tokenizer):
    def encode(self, text: str) -> list[int]:
        return list(range(1, len(text) + 1))

    def decode(self, tokens: list[int]) -> str:
        return "".join(str(token) for token in tokens)

    def get_end_token(self) -> int:
        return 0


@pytest.fixture
def three_text_dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw_text"
    path = root / "CulturaX" / "dataset_file.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "text": [
            "This is the first text.",
            "This is the second text.",
            "This is the third text.",
        ]
    }).to_parquet(path, engine="auto", index=False)
    return root


@pytest.fixture
def sequential_dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw_text"
    path = root / "CulturaX" / "dataset_file.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"text": ["abcdefghij"]}).to_parquet(path, engine="auto", index=False)
    return root


@pytest.fixture
def token_count_dataset_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw_text"
    path = root / "CulturaX" / "dataset_file.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"text": ["abc", "def", "ghi"]}).to_parquet(path, engine="auto", index=False)
    return root


def test_load_text_from_parquet_file(three_text_dataset_dir: Path):
    parquet_path = three_text_dataset_dir / "CulturaX" / "dataset_file.parquet"
    raw_texts = load_text_from_parquet_file(parquet_path, "text")
    assert len(raw_texts) > 0


def test_pre_training_dataset(three_text_dataset_dir: Path):
    pre_training_dataset = PreTrainingDataset(
        str(three_text_dataset_dir),
        sequence_length=6,
        train_split_size=0.67,
        batch_size=1,
        tokenizer=TiktokenTokenizer(),
        device=torch.device("cpu"),
    )

    dataset = []
    x, y = pre_training_dataset.get_batch("train")
    dataset.append((x, y))
    x, y = pre_training_dataset.get_batch("val")
    dataset.append((x, y))

    for x, y in dataset:
        assert len(x[0]) == len(y[0])
        assert x.dtype == torch.long and y.dtype == torch.long
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_sequential_batch_advances_by_sequence_length(sequential_dataset_dir: Path):
    pre_training_dataset = PreTrainingDataset(
        str(sequential_dataset_dir),
        sequence_length=3,
        train_split_size=1.0,
        batch_size=2,
        tokenizer=LengthTokenizer(),
        device=torch.device("cpu"),
    )

    x, y = pre_training_dataset.get_sequential_batch("train")

    assert torch.equal(x, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert torch.equal(y, torch.tensor([[2, 3, 4], [5, 6, 7]]))


def test_count_pre_training_tokens_matches_dataset_split(token_count_dataset_dir: Path):
    counts = count_pre_training_tokens(
        str(token_count_dataset_dir),
        train_split_size=2 / 3,
        tokenizer=LengthTokenizer(),
        random_seed=0,
    )

    assert counts.train_tokens == 8
    assert counts.val_tokens == 4
