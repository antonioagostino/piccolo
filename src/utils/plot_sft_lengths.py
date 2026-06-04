"""
Plot the token-length distribution of the SFT dataset.

Each sample is formatted as alternating Human / GPT turns (the same layout
that the tokenisation script will use) and tokenised with the model's tiktoken
encoder.  A vertical line marks the model's sequence-length limit so you can
immediately see what fraction of samples would be truncated.

Usage
-----
    python -m src.utils.plot_sft_lengths
    python -m src.utils.plot_sft_lengths \\
        --data-file data/conversational/alpaca-gpt4-italian/alpaca-gpt4-italian.json \\
        --sequence-length 2048 \\
        --encoding cl100k_base \\
        --output-file data/conversational/length_distribution.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from tqdm.auto import tqdm  # type: ignore[import-untyped]


# ──────────────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_sample(conversations: list[dict]) -> str:
    """
    Render a ShareGPT-style conversation as a plain-text string.

    Each turn becomes ``"Human: <text>\\n"`` or ``"GPT: <text>\\n"`` depending
    on the ``from`` field.  Unknown roles are kept as-is.

    Args:
        conversations: List of turn dicts with ``"from"`` and ``"value"`` keys.

    Returns:
        str: The full formatted conversation.
    """
    role_map = {"human": "Human", "gpt": "Model"}
    lines = []
    for turn in conversations:
        role = role_map.get(turn["from"], turn["from"])
        lines.append(f"{role}: {turn['value'].strip()}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Length computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_lengths(data: list[dict], enc: tiktoken.Encoding) -> np.ndarray:
    """
    Tokenise every sample and return an array of token counts.

    Args:
        data: List of dataset records, each with a ``"conversations"`` field.
        enc:  Tiktoken encoder to use.

    Returns:
        np.ndarray: Integer array of shape ``(len(data),)`` with per-sample
            token counts.
    """
    lengths = []
    for sample in tqdm(data, desc="tokenising", unit=" samples"):
        text = format_sample(sample["conversations"])
        lengths.append(len(enc.encode(text)))
    return np.array(lengths, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_distribution(
    lengths: np.ndarray,
    sequence_length: int,
    output_file: Path | None,
) -> None:
    """
    Plot a histogram of token lengths with a vertical line at the sequence
    length limit and save (or display) the figure.

    Args:
        lengths:         Array of per-sample token counts.
        sequence_length: Model context window; drawn as a vertical limit line.
        output_file:     Path to save the PNG, or None to show interactively.
    """
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    n_total  = len(lengths)
    n_over   = int((lengths > sequence_length).sum())
    pct_over = 100 * n_over / n_total

    percentiles = np.percentile(lengths, [50, 90, 95, 99])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(lengths, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)

    ax.axvline(sequence_length, color="#DD4444", linewidth=1.8, linestyle="--",
               label=f"max seq len = {sequence_length}")

    # Shade the region beyond the limit
    ax.axvspan(sequence_length, lengths.max() * 1.02,
               alpha=0.12, color="#DD4444", label=f"{pct_over:.1f}% truncated")

    ax.set_xlabel("Token length", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_title("SFT dataset — token length distribution", fontsize=14)
    ax.legend(fontsize=11)

    stats_text = (
        f"n = {n_total:,}\n"
        f"median = {percentiles[0]:.0f}\n"
        f"p90 = {percentiles[1]:.0f}\n"
        f"p95 = {percentiles[2]:.0f}\n"
        f"p99 = {percentiles[3]:.0f}\n"
        f"max = {lengths.max()}"
    )
    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    plt.tight_layout()

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved → {output_file}")
    else:
        plt.show()

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    print(f"Loading {args.data_file} …")
    data = json.loads(args.data_file.read_text(encoding="utf-8"))
    print(f"  {len(data):,} samples loaded")

    enc = tiktoken.get_encoding(args.encoding)
    lengths = compute_lengths(data, enc)

    n_over   = int((lengths > args.sequence_length).sum())
    pct_over = 100 * n_over / len(lengths)

    print(f"\n── Length statistics ──────────────────────────")
    print(f"  samples           : {len(lengths):>8,}")
    print(f"  min               : {lengths.min():>8,} tokens")
    print(f"  median            : {int(np.median(lengths)):>8,} tokens")
    print(f"  mean              : {lengths.mean():>8.1f} tokens")
    print(f"  p90               : {int(np.percentile(lengths, 90)):>8,} tokens")
    print(f"  p95               : {int(np.percentile(lengths, 95)):>8,} tokens")
    print(f"  p99               : {int(np.percentile(lengths, 99)):>8,} tokens")
    print(f"  max               : {lengths.max():>8,} tokens")
    print(f"  ── limit = {args.sequence_length} ──────────────────────")
    print(f"  above limit       : {n_over:>8,}  ({pct_over:.1f}%)")
    print(f"  within limit      : {len(lengths) - n_over:>8,}  ({100 - pct_over:.1f}%)")

    plot_distribution(lengths, args.sequence_length, args.output_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token-length distribution of the SFT dataset."
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
        help="Model context window (vertical limit line). Default: 2048.",
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
        help="Save the plot to this path instead of displaying it. "
             "Example: data/conversational/length_distribution.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
