"""
Tokenize the SFT dataset for fine-tuning.

Reads the ShareGPT-style JSON produced by download_sft_dataset, applies a
length policy, tokenises each sample and writes compact binary files that
FinetuneDataset can read directly.

Conversation format
-------------------
Each conversation is rendered as alternating "Human:" / "GPT:" lines:

    Human: <instruction>
    GPT: <response><EOS>

Length policy
-------------
Let *pct_exceed* = fraction of samples whose token length exceeds
*sequence_length*.

  pct_exceed ≤ 20 %  →  keep every sample; truncate the ones that exceed.
  pct_exceed > 20 %  →  sort exceeding samples longest-first; discard enough
                         of them (the longest) until only 20 % of the total
                         remain above the limit; truncate those survivors.

Truncation always shortens the Human turn first.  The GPT response is only
touched if the Human turn alone still cannot fit.

Output layout (--output-dir)
-----------------------------
    train.bin            flat uint32 — all train tokens concatenated
    train_offsets.npy    int64 (n_train + 1,) — start index of each sample
    val.bin
    val_offsets.npy
    metadata.json

Usage
-----
    python -m src.utils.tokenize_sft_dataset \\
        --input  data/conversational/alpaca-gpt4-italian/alpaca-gpt4-italian.json \\
        --output-dir data/finetune \\
        --sequence-length 2048 \\
        --val-split 0.1 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tiktoken
from tqdm.auto import tqdm  # type: ignore[import-untyped]

# ──────────────────────────────────────────────────────────────────────────────
# Turn formatting
# ──────────────────────────────────────────────────────────────────────────────

_HUMAN_HEADER = "Human: "
_GPT_HEADER   = "Model: "
_TURN_SEP     = "\n"

MAX_EXCEED_FRACTION = 0.20   # policy threshold


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenizedTurn:
    role: str               # "human" or "gpt"
    header_ids: list[int]
    text_ids: list[int]

    def length(self) -> int:
        return len(self.header_ids) + len(self.text_ids)


@dataclass
class TokenizedSample:
    turns: list[TokenizedTurn]
    sep_ids: list[int]      # tokens for "\n" between turns
    eos_id: int

    def raw_length(self) -> int:
        """Total tokens before any truncation."""
        total = sum(t.length() for t in self.turns)
        total += len(self.sep_ids) * max(0, len(self.turns) - 1)
        total += 1  # EOS
        return total

    def to_tokens(self, max_length: int | None = None) -> list[int]:
        """
        Materialise the sample as a flat token list.

        If *max_length* is given and the sample is too long, human turns are
        shortened first (longest first).  GPT turns are only shortened if
        reducing all human text to a single token is still not enough.
        """
        turns = [TokenizedTurn(t.role, list(t.header_ids), list(t.text_ids))
                 for t in self.turns]

        if max_length is not None:
            def _total() -> int:
                t = sum(t.length() for t in turns)
                t += len(self.sep_ids) * max(0, len(turns) - 1)
                t += 1  # EOS
                return t

            # Shorten human turns first (in order: first turn, then subsequent)
            for t in turns:
                if t.role != "human":
                    continue
                if _total() <= max_length:
                    break
                excess = _total() - max_length
                new_len = max(1, len(t.text_ids) - excess)
                t.text_ids = t.text_ids[:new_len]

            # Fallback: shorten GPT turns if human text is already at 1 token
            for t in turns:
                if t.role != "gpt":
                    continue
                if _total() <= max_length:
                    break
                excess = _total() - max_length
                new_len = max(1, len(t.text_ids) - excess)
                t.text_ids = t.text_ids[:new_len]

        tokens: list[int] = []
        for i, t in enumerate(turns):
            if i > 0:
                tokens.extend(self.sep_ids)
            tokens.extend(t.header_ids)
            tokens.extend(t.text_ids)
        tokens.append(self.eos_id)
        return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_sample(
    conversations: list[dict],
    enc: tiktoken.Encoding,
    eos_id: int,
) -> TokenizedSample:
    sep_ids = enc.encode(_TURN_SEP)
    turns: list[TokenizedTurn] = []
    for turn in conversations:
        role = turn["from"]
        header = _HUMAN_HEADER if role == "human" else _GPT_HEADER
        turns.append(TokenizedTurn(
            role=role,
            header_ids=enc.encode(header),
            text_ids=enc.encode(turn["value"].strip()),
        ))
    return TokenizedSample(turns=turns, sep_ids=sep_ids, eos_id=eos_id)


# ──────────────────────────────────────────────────────────────────────────────
# Length policy
# ──────────────────────────────────────────────────────────────────────────────

def apply_length_policy(
    samples: list[TokenizedSample],
    sequence_length: int,
) -> list[TokenizedSample]:
    """
    Apply the 20 % length policy and return the samples to keep.

    Samples returned may still have raw_length() > sequence_length; callers
    must pass max_length to to_tokens() to materialise them correctly.
    """
    n_total   = len(samples)
    exceeding = [s for s in samples if s.raw_length() > sequence_length]
    within    = [s for s in samples if s.raw_length() <= sequence_length]

    pct = len(exceeding) / n_total
    print(f"  {len(exceeding):,} / {n_total:,} samples exceed {sequence_length} tokens "
          f"({pct * 100:.1f} %)")

    if pct <= MAX_EXCEED_FRACTION:
        print(f"  ≤ {MAX_EXCEED_FRACTION*100:.0f} % — keeping all, will truncate the "
              f"{len(exceeding)} that exceed")
        return samples

    # Sort longest-first so we discard the most extreme outliers
    exceeding.sort(key=lambda s: s.raw_length(), reverse=True)
    max_allowed = math.floor(MAX_EXCEED_FRACTION * n_total)
    n_discard   = len(exceeding) - max_allowed
    discarded   = exceeding[:n_discard]
    to_truncate = exceeding[n_discard:]

    print(f"  > {MAX_EXCEED_FRACTION*100:.0f} % — discarding {n_discard} longest samples, "
          f"truncating {len(to_truncate)} survivors")
    if discarded:
        lens = [s.raw_length() for s in discarded]
        print(f"    discarded length range: {min(lens)} – {max(lens)} tokens")

    return within + to_truncate


# ──────────────────────────────────────────────────────────────────────────────
# Binary writer (same layout as FinetuneDataset expects)
# ──────────────────────────────────────────────────────────────────────────────

def _write_split(
    samples: list[TokenizedSample],
    sequence_length: int,
    output_dir: Path,
    name: str,
) -> dict:
    if not samples:
        raise ValueError(f"Split '{name}' has no samples.")

    chunks: list[list[int]] = [
        s.to_tokens(max_length=sequence_length) for s in samples
    ]

    flat    = np.concatenate([np.array(c, dtype=np.uint32) for c in chunks])
    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    for i, c in enumerate(chunks):
        offsets[i + 1] = offsets[i] + len(c)

    flat.tofile(output_dir / f"{name}.bin")
    np.save(output_dir / f"{name}_offsets.npy", offsets)

    lengths = [len(c) for c in chunks]
    return {
        "n_samples":   len(chunks),
        "n_tokens":    int(flat.size),
        "min_length":  int(min(lengths)),
        "max_length":  int(max(lengths)),
        "mean_length": round(float(np.mean(lengths)), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_sft_dataset(
    input_path: Path,
    output_dir: Path,
    sequence_length: int,
    val_split: float,
    seed: int,
    encoding: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enc    = tiktoken.get_encoding(encoding)
    eos_id = enc.eot_token

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {input_path} …")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"  {len(data):,} samples")

    # ── Tokenize ─────────────────────────────────────────────────────────────
    print("Tokenising …")
    samples: list[TokenizedSample] = [
        _tokenize_sample(item["conversations"], enc, eos_id)
        for item in tqdm(data, unit=" samples")
    ]

    # ── Length policy ─────────────────────────────────────────────────────────
    print(f"\nApplying length policy (limit = {sequence_length} tokens) …")
    samples = apply_length_policy(samples, sequence_length)
    print(f"  → {len(samples):,} samples retained")

    # ── Shuffle and split ────────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val       = max(1, int(len(samples) * val_split))
    val_samples  = samples[:n_val]
    train_samples = samples[n_val:]

    # ── Write ────────────────────────────────────────────────────────────────
    print(f"\nWriting output → {output_dir}/")
    train_stats = _write_split(train_samples, sequence_length, output_dir, "train")
    val_stats   = _write_split(val_samples,   sequence_length, output_dir, "val")

    metadata = {
        "source":          str(input_path),
        "encoding":        encoding,
        "eos_token_id":    eos_id,
        "sequence_length": sequence_length,
        "val_split":       val_split,
        "seed":            seed,
        "train":           train_stats,
        "val":             val_stats,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"  train : {train_stats['n_samples']:>6,} samples  "
          f"{train_stats['n_tokens']:>10,} tokens  "
          f"len {train_stats['min_length']}–{train_stats['max_length']}")
    print(f"  val   : {val_stats['n_samples']:>6,} samples  "
          f"{val_stats['n_tokens']:>10,} tokens  "
          f"len {val_stats['min_length']}–{val_stats['max_length']}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize the SFT dataset for fine-tuning."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/conversational/alpaca-gpt4-italian/alpaca-gpt4-italian.json"),
        help="Path to the downloaded JSON dataset file.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/finetune"),
        help="Directory where binary output files are written (default: data/finetune).",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=2048,
        help="Model context length (default: 2048).",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of samples assigned to validation (default: 0.1).",
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
    args = _parse_args()
    tokenize_sft_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        val_split=args.val_split,
        seed=args.seed,
        encoding=args.encoding,
    )
