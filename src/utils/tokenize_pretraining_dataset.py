import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.tokenizer import TiktokenTokenizer


def load_text_from_parquet_file(dataset_file_path: Path,
                                raw_texts_column_name: str) -> list[str]:
    """Load raw texts from a single Parquet file."""
    return pd.read_parquet(dataset_file_path)[raw_texts_column_name].to_list()

PRE_TRAINING_DATASETS_EXTENSIONS = {
    "CulturaX": ".parquet"
}
PRE_TRAINING_DATASETS_FN = {
    "CulturaX": load_text_from_parquet_file
}
PRE_TRAINING_DATASETS_ARGS = {
    "CulturaX": ["text"]
}
CHUNK_TOKENS = 2_000_000  # flush to disk every around 8 MB

def get_pre_training_dataset_files(data_path: Path,
                                   rng: random.Random) -> list[Path]:
    """Retrieve and return all the files from pre-training datasets."""
    pre_training_datasets_dirs = [ds_dir for ds_dir in Path(data_path).iterdir() if ds_dir.is_dir()]

    # Shuffle to avoid bias coming from files and dirs ordering
    rng.shuffle(pre_training_datasets_dirs)

    all_files: list[Path] = []
    for pre_training_dataset_dir in pre_training_datasets_dirs:
        pre_training_files = [file for file in pre_training_dataset_dir.iterdir() if file.is_file()]
        # Shuffle to avoid bias coming from files and dirs ordering
        rng.shuffle(pre_training_files)
        all_files.extend(pre_training_files)

    return all_files

def validate_pre_training_file(file: Path) -> None:
    """Validate that a data file has the expected extension for its dataset."""
    dataset_name = file.parent.stem
    expected_extension = PRE_TRAINING_DATASETS_EXTENSIONS[dataset_name]
    if file.suffix != expected_extension:
        raise ValueError(f"Pretraining file's extension under directory '{dataset_name}' do not match the\
                        correct extension mapping: {expected_extension}")
    

def load_pre_training_raw_texts(file: Path) -> list[str]:
    """Load raw texts from a pre-training data file using the mapped loader."""
    dataset_name = file.parent.stem
    return PRE_TRAINING_DATASETS_FN[dataset_name](
        file,
        *PRE_TRAINING_DATASETS_ARGS[dataset_name]
    )

def flush(buf: list[int], fh) -> None:
    if buf:
        np.array(buf, dtype=np.uint32).tofile(fh)
        buf.clear()


def tokenize_dataset(
    data_dir: Path,
    output_dir: Path,
    encoding: str,
    train_split: float,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = TiktokenTokenizer(encoding)
    eos = tokenizer.get_end_token()
    rng = random.Random(seed)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    train_tokens = 0
    val_tokens = 0
    files_processed = 0

    t_start = time.perf_counter()

    with open(train_path, "wb") as train_fh, open(val_path, "wb") as val_fh:
        train_buf: list[int] = []
        val_buf:   list[int] = []

        for file in tqdm(get_pre_training_dataset_files(data_dir, rng), desc="Tokenizing", unit="file"):
            validate_pre_training_file(file)
            raw_texts = load_pre_training_raw_texts(file)
            rng.shuffle(raw_texts)

            split_idx = int(len(raw_texts) * train_split)
            train_texts = raw_texts[:split_idx]
            val_texts = raw_texts[split_idx:]

            for text in train_texts:
                ids = tokenizer.encode(text)
                ids.append(eos)
                train_buf.extend(ids)
                train_tokens += len(ids)
                if len(train_buf) >= CHUNK_TOKENS:
                    flush(train_buf, train_fh)

            for text in val_texts:
                ids = tokenizer.encode(text)
                ids.append(eos)
                val_buf.extend(ids)
                val_tokens += len(ids)
                if len(val_buf) >= CHUNK_TOKENS:
                    flush(val_buf, val_fh)

            files_processed += 1

        flush(train_buf, train_fh)
        flush(val_buf,   val_fh)

    elapsed = time.perf_counter() - t_start

    metadata = {
        "encoding":      encoding,
        "dtype":         "uint32",
        "train_split":   train_split,
        "seed":          seed,
        "files":         files_processed,
        "train_tokens":  train_tokens,
        "val_tokens":    val_tokens,
        "total_tokens":  train_tokens + val_tokens,
        "train_bin":     str(train_path),
        "val_bin":       str(val_path),
        "elapsed_s":     round(elapsed, 1),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Tokenization complete!")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize raw pre-training data into binary files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw_text"),
        help="Root directory containing dataset sub-directories (e.g. CulturaX/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_text_tokenized"),
        help="Directory where train.bin, val.bin, and metadata.json are written.",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="Tiktoken encoding name (must match the one used during training).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of texts assigned to the train split (per file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling file and text shuffle order.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tokenize_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        encoding=args.encoding,
        train_split=args.train_split,
        seed=args.seed,
    )
