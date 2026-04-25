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

@pytest.fixture(scope="module")
def parquet_dataset_file() -> Path:
    datasets_root_dir = Path("test/fixtures/raw_text/")
    path = datasets_root_dir / "CulturaX" / "dataset_file.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "text": [
            "This is the first text.",
            "This is the second text.",
            "This is the third text."
        ]
    })
    df.to_parquet(path, engine="auto", index=False)
    return datasets_root_dir

def test_load_text_from_parquet_file(parquet_dataset_file):
    dataset_file_path = Path(parquet_dataset_file) / "CulturaX" / "dataset_file.parquet"
    raw_texts = load_text_from_parquet_file(dataset_file_path,
                                            "text")
    assert len(raw_texts) > 0

def test_pre_training_dataset(parquet_dataset_file):
    tokenizer = TiktokenTokenizer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sequence_length = 6
    train_split = 0.67
    batch_size = 1
    pre_training_dataset = PreTrainingDataset(parquet_dataset_file,
                                              sequence_length,
                                              train_split,
                                              batch_size,
                                              tokenizer,
                                              device
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


def test_sequential_batch_advances_by_sequence_length(tmp_path: Path):
    datasets_root_dir = tmp_path / "raw_text"
    path = datasets_root_dir / "CulturaX" / "dataset_file.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"text": ["abcdefghij"]})
    df.to_parquet(path, engine="auto", index=False)
    pre_training_dataset = PreTrainingDataset(
        str(datasets_root_dir),
        sequence_length=3,
        train_split_size=1.0,
        batch_size=2,
        tokenizer=LengthTokenizer(),
        device=torch.device("cpu")
    )

    x, y = pre_training_dataset.get_sequential_batch("train")

    assert torch.equal(x, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert torch.equal(y, torch.tensor([[2, 3, 4], [5, 6, 7]]))


def test_count_pre_training_tokens_matches_dataset_split(tmp_path: Path):
    datasets_root_dir = tmp_path / "raw_text"
    path = datasets_root_dir / "CulturaX" / "dataset_file.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"text": ["abc", "def", "ghi"]})
    df.to_parquet(path, engine="auto", index=False)

    counts = count_pre_training_tokens(
        str(datasets_root_dir),
        train_split_size=2 / 3,
        tokenizer=LengthTokenizer(),
        random_seed=0,
    )

    assert counts.train_tokens == 8
    assert counts.val_tokens == 4
