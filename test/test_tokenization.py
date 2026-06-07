import pytest
import pandas as pd
from pathlib import Path
import torch
import numpy as np
from src.utils.tokenize_pretraining_dataset import tokenize_dataset
from src.tokenizer import TiktokenTokenizer

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

def test_tokenize_pretraining_dataset(three_text_dataset_dir: Path):
    tokenizer_encoding = "cl100k_base"
    tokenizer = TiktokenTokenizer(tokenizer_encoding)
    tokenized_data_dir = three_text_dataset_dir.parent / "raw_text_tokenized"
    tokenize_dataset(three_text_dataset_dir,
                     tokenized_data_dir,
                     tokenizer_encoding,
                     0.9,
                     42)
    train_tokenized_text = np.memmap(tokenized_data_dir / "train.bin", dtype=np.uint32, mode="r")
    val_tokenized_text = np.memmap(tokenized_data_dir / "val.bin", dtype=np.uint32, mode="r")
    train = np.array(train_tokenized_text, dtype=np.int64)
    val = np.array(val_tokenized_text, dtype=np.int64)
    expected_raw_train_text = "This is the second text.<|endoftext|>This is the first text.<|endoftext|>"
    expected_raw_val_text = "This is the third text.<|endoftext|>"

    assert tokenizer.decode(train) == expected_raw_train_text
    assert tokenizer.decode(val) == expected_raw_val_text