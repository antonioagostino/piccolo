import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from src.tokenizer import TiktokenTokenizer

def write_split(
    chunks: list[list[int]],
    output_dir: Path,
    split_type: str,
) -> dict:
    flat = np.concatenate([np.array(c, dtype=np.uint32) for c in chunks])
    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    for i, c in enumerate(chunks):
        offsets[i + 1] = offsets[i] + len(c)

    flat.tofile(output_dir / f"{split_type}.bin")
    np.save(output_dir / f"{split_type}_offsets.npy", offsets)

    lengths = [len(c) for c in chunks]
    return {
        "n_samples": len(chunks),
        "n_tokens": int(flat.size),
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "mean_length": round(float(np.mean(lengths)), 1),
    }

def tokenize_sft_dataset(
    input_path: Path,
    output_dir: Path,
    sequence_length: int,
    train_split: float,
    seed: int,
    encoding: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TiktokenTokenizer(encoding)
    eos_id = tokenizer.get_end_token()
    sft_data = json.loads(input_path.read_text(encoding="utf-8"))
    samples: list[list[int]] = []

    for sample in tqdm(sft_data, unit=" samples"):
        turns: list[int] = []
        n_turns = len(sample["conversations"])
        for n_turn, turn in enumerate(sample["conversations"]):
            sender = "Human: " if turn["from"] == "human" else "Model: "
            message = turn["value"].strip()
            tokenized_turn = tokenizer.encode(sender + message)
            last_turn = n_turn == n_turns - 1
            sep = [eos_id] if last_turn else tokenizer.encode("\n")
            tokenized_turn.extend(sep)
            turns.extend(tokenized_turn)

        samples.append(turns)

    rng = random.Random(seed)
    rng.shuffle(samples)
    train_split_index = int(len(samples) * train_split)
    train_samples  = samples[:train_split_index]
    val_samples = samples[train_split_index:]

    train_stats = write_split(train_samples, output_dir, "train")
    val_stats = write_split(val_samples, output_dir, "val")

    metadata = {
        "source": str(input_path),
        "encoding": encoding,
        "eos_token_id": eos_id,
        "sequence_length": sequence_length,
        "train_split": train_split,
        "seed": seed,
        "train": train_stats,
        "val": val_stats,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Train : {train_stats['n_samples']:>6,} samples  "
          f"{train_stats['n_tokens']:>10,} tokens  "
          f"len {train_stats['min_length']}–{train_stats['max_length']}")
    print(f"Val   : {val_stats['n_samples']:>6,} samples  "
          f"{val_stats['n_tokens']:>10,} tokens  "
          f"len {val_stats['min_length']}–{val_stats['max_length']}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize the SFT dataset for fine-tuning."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/conversational/alpaca-gpt4-italian/alpaca-gpt4-italian.json"),
        help="Path to the downloaded JSON dataset file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/conversational_tokenized"),
        help="Directory where binary output files are written (default: data/conversational_tokenized).",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=2048,
        help="Model context length (default: 2048).",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.9,
        help="Fraction of samples assigned to the training set (default: 0.9).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42).",
    )
    parser.add_argument(
        "--encoding", type=str, default="cl100k_base",
        help="Tiktoken encoding name (default: cl100k_base).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenize_sft_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        train_split=args.train_split,
        seed=args.seed,
        encoding=args.encoding,
    )
