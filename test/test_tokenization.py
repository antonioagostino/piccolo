import pytest
import pandas as pd
from pathlib import Path
import json
import numpy as np
from src.utils.tokenize_pretraining_dataset import tokenize_dataset
from src.utils.tokenize_sft_dataset import tokenize_sft_dataset
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

@pytest.fixture
def sft_mock_dataset_file(tmp_path: Path) -> Path:
    root = tmp_path / "conversational"
    path = root / "alpaca-gpt4-italian" / "alpaca-gpt4-italian.json"
    path.parent.mkdir(parents=True)
    mock_sft_data = [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "Test per SFT n1.\n"
                },
                {
                    "from": "gpt",
                    "value": "Risposta al test n1"
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "Test per SFT n2.\n"
                },
                {
                    "from": "gpt",
                    "value": "Risposta al test n2"
                }
            ]
        },
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "Test per SFT n3.\n"
                },
                {
                    "from": "gpt",
                    "value": "Risposta al test n3"
                }
            ]
        }
    ]

    with open(path, "w") as f:
        f.write(json.dumps(mock_sft_data, indent=2))

    return path

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

def test_tokenize_sftdataset(sft_mock_dataset_file: Path):
    output_dir = sft_mock_dataset_file.parent.parent.parent / "conversational_tokenized"
    tokenizer = TiktokenTokenizer("cl100k_base")
    tokenize_sft_dataset(sft_mock_dataset_file,
                         output_dir,
                         16,
                         0.9,
                         42,
                         "cl100k_base")
    
    train_tokenized_text = np.memmap(output_dir / "train.bin", dtype=np.uint32, mode="r")
    val_tokenized_text = np.memmap(output_dir / "val.bin", dtype=np.uint32, mode="r")
    train_offsets = np.load(output_dir / "train_offsets.npy")
    val_offsets = np.load(output_dir / "val_offsets.npy")
    train_expected_samples = [
        "Human: Test per SFT n2.\nModel: Risposta al test n2<|endoftext|>",
        "Human: Test per SFT n1.\nModel: Risposta al test n1<|endoftext|>",
    ]

    val_expected_samples = [
        "Human: Test per SFT n3.\nModel: Risposta al test n3<|endoftext|>",
    ]
    offset = 0
    for exp_train_sample in train_expected_samples:
        train_sample = np.array(train_tokenized_text[train_offsets[offset]:train_offsets[offset + 1]], dtype=np.int64)
        assert tokenizer.decode(train_sample) == exp_train_sample
        offset += 1
    offset = 0
    for exp_val_sample in val_expected_samples:
        val_sample = np.array(val_tokenized_text[val_offsets[offset]:val_offsets[offset + 1]], dtype=np.int64)
        assert tokenizer.decode(val_sample) == exp_val_sample
        offset += 1