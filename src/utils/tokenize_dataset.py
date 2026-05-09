"""
Pre-tokenize the raw pre-training data and write flat binary files that can
be read with numpy.memmap during training.

Output layout
─────────────
  <output_dir>/train.bin   – uint32 token IDs, flat, row-major
  <output_dir>/val.bin     – uint32 token IDs, flat, row-major
  <output_dir>/metadata.json

The binary files contain nothing but a contiguous stream of uint32 values;
any numpy.memmap(path, dtype="uint32", mode="r") call will reconstruct the
full token array without any header parsing.

The shuffle + per-file train/val split logic deliberately mirrors
PreTrainingDataset so the resulting distribution is identical to the
existing streaming path.

Usage
─────
    python -m src.utils.tokenize_dataset \\
        --data-dir   ./data/raw_text \\
        --output-dir ./data/tokenized \\
        --encoding   cl100k_base \\
        --train-split 0.9 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.dataset import (
    get_pre_training_dataset_files,
    load_pre_training_raw_texts,
    validate_pre_training_file,
)
from src.tokenizer import TiktokenTokenizer

# uint32 covers vocab sizes up to 4 294 967 295; cl100k_base vocab is 100 277.
TOKEN_DTYPE = np.uint32
CHUNK_TOKENS = 2_000_000  # flush to disk every ~8 MB


def _flush(buf: list[int], fh) -> None:
    if buf:
        np.array(buf, dtype=TOKEN_DTYPE).tofile(fh)
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
    val_path   = output_dir / "val.bin"

    train_tokens = 0
    val_tokens   = 0
    files_processed = 0

    t_start = time.perf_counter()

    with open(train_path, "wb") as train_fh, open(val_path, "wb") as val_fh:
        train_buf: list[int] = []
        val_buf:   list[int] = []

        all_files = list(get_pre_training_dataset_files(data_dir, rng))
        for file in tqdm(all_files, desc="tokenizing", unit="file"):
            validate_pre_training_file(file)
            raw_texts = load_pre_training_raw_texts(file)
            rng.shuffle(raw_texts)

            split_idx = int(len(raw_texts) * train_split)
            train_texts = raw_texts[:split_idx]
            val_texts   = raw_texts[split_idx:]

            for text in train_texts:
                ids = tokenizer.encode(text)
                ids.append(eos)
                train_buf.extend(ids)
                train_tokens += len(ids)
                if len(train_buf) >= CHUNK_TOKENS:
                    _flush(train_buf, train_fh)

            for text in val_texts:
                ids = tokenizer.encode(text)
                ids.append(eos)
                val_buf.extend(ids)
                val_tokens += len(ids)
                if len(val_buf) >= CHUNK_TOKENS:
                    _flush(val_buf, val_fh)

            files_processed += 1

        _flush(train_buf, train_fh)
        _flush(val_buf,   val_fh)

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

    gb = 1 << 30
    print()
    print("=" * 56)
    print("  TOKENIZATION COMPLETE")
    print("=" * 56)
    print(f"  Files processed : {files_processed}")
    print(f"  Train tokens    : {train_tokens:,}  "
          f"({train_path.stat().st_size / gb:.3f} GB)")
    print(f"  Val tokens      : {val_tokens:,}  "
          f"({val_path.stat().st_size / gb:.3f} GB)")
    print(f"  Elapsed         : {elapsed:.0f} s")
    print(f"  Speed           : {(train_tokens + val_tokens) / elapsed:,.0f} tok/s")
    print(f"  Metadata        : {output_dir / 'metadata.json'}")
    print("=" * 56)
    print()


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
        default=Path("data/tokenized"),
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
