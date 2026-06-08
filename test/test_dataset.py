import pytest
from pathlib import Path
import json

import pandas as pd
import torch

from src.dataset import TokenizedPreTrainingDataset, TokenizedFinetuneDataset
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

def test_pretraining_dataset(three_text_dataset_dir: Path):
    tokenizer_encoding = "cl100k_base"
    tokenized_data_dir = three_text_dataset_dir.parent / "raw_text_tokenized"
    tokenize_dataset(three_text_dataset_dir,
                     tokenized_data_dir,
                     tokenizer_encoding,
                     0.9,
                     42)
    pretraining_dataset = TokenizedPreTrainingDataset(
        tokenized_data_dir,
        6,
        1
    )

    x, y = pretraining_dataset.get_sequential_batch("train")
    assert torch.all(x[0][1:] == y[0][:-1])
    x, y = pretraining_dataset.get_sequential_batch("val")
    assert torch.all(x[0][1:] == y[0][:-1])
    
def test_sft_dataset(sft_mock_dataset_file: Path):
    output_dir = sft_mock_dataset_file.parent.parent.parent / "conversational_tokenized"
    sequence_length = 16
    train_split = 0.9
    seed = 42
    tok_encoding = "cl100k_base"
    batch_size = 1
    tokenizer = TiktokenTokenizer(tok_encoding)
    tokenize_sft_dataset(sft_mock_dataset_file,
                         output_dir,
                         sequence_length,
                         train_split,
                         seed,
                         tok_encoding)
    
    tokenized_sft_dataset = TokenizedFinetuneDataset(
        output_dir,
        sequence_length,
        batch_size,
        tokenizer.get_end_token(),
        seed
    )
    x, y = tokenized_sft_dataset.get_sequential_batch("train")
    assert torch.all(x[0][1:] == y[0][:-1])
    x, y = tokenized_sft_dataset.get_sequential_batch("val")
    assert torch.all(x[0][1:] == y[0][:-1])