import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from tqdm.auto import tqdm  # type: ignore[import-untyped]

def format_sample(conversations: list[dict]) -> str:
    """Transform a ShareGPT-style conversation as a plain-text string."""
    lines = []
    for turn in conversations:
        role = turn["from"]
        lines.append(f"{role}: {turn['value'].strip()}")
    return "\n".join(lines)

def main() -> None:
    args = parse_args()
    assert args.data_file.exists() and args.data_file.suffix == ".json", "Invalid dataset file. Alpaca GPT4 dataset's \
                                                                            is encoded as JSON file."
    data = json.loads(args.data_file.read_text(encoding="utf-8"))
    enc = tiktoken.get_encoding(args.encoding)
    seqs_lengths = []
    for sample in tqdm(data, desc="Tokenising", unit=" samples"):
        text = format_sample(sample["conversations"])
        seqs_lengths.append(len(enc.encode(text)))
    
    seqs_lengths_arr = np.array(seqs_lengths, dtype=np.int64)

    n_over   = int((seqs_lengths_arr > args.sequence_length).sum())
    pct_over = 100 * n_over / len(seqs_lengths_arr)

    print(f"SFT sequences lengths statistics")
    print(f"Samples: {len(seqs_lengths_arr)}")
    print(f"Min: {seqs_lengths_arr.min()} tokens")
    print(f"Median: {int(np.median(seqs_lengths_arr))} tokens")
    print(f"Max: {seqs_lengths_arr.max()} tokens")
    print(f"Above limit: {n_over}  ({pct_over:.1f}%)")
    print(f"Within limit: {len(seqs_lengths_arr) - n_over}  ({100 - pct_over:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token-length distribution of the SFT dataset's sequences."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(
            "data/conversational/alpaca-gpt4-italian/alpaca-gpt4-italian.json"
        ),
        help="Path to the downloaded JSON dataset file.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Model context window. Default: 2048.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="Tiktoken encoding name (default: cl100k_base).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Save the plot to this path instead of displaying it."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
