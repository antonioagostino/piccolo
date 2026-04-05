import pytest
import pandas as pd
from pathlib import Path
import torch
from src.dataset import PreTrainingDataset, load_text_from_parquet_file
from src.tokenizer import TiktokenTokenizer

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
