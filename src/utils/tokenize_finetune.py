"""
Tokenize a WhatsApp chat export for finetuning.

Parses and cleans the chat, segments it into conversation sessions, tokenizes
each session, and writes compact binary files readable by FinetuneDataset.

Output layout (--output-dir)
-----------------------------
  train.bin          flat uint32 array — all train sample tokens concatenated
  train_offsets.npy  int64 array of shape (n_train + 1,): train_offsets[i] is
                     the start index of sample i in train.bin
  val.bin            same for validation
  val_offsets.npy
  metadata.json      counts, lengths, session gap, encoding, …

Session → token mapping
------------------------
Each session is formatted as "Sender: text\\n" lines, then tokenised with the
tiktoken encoder.  An EOS token is appended so the model learns where sessions
end.  Sessions that exceed --sequence-length tokens are split at the nearest
message boundary, keeping each chunk ≤ sequence_length tokens.

Usage
-----
    python -m src.utils.tokenize_finetune \\
        --input  ./data/conversational/lorenzo.txt \\
        --output-dir ./data/finetune \\
        --session-gap 30 \\
        --val-split 0.1 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.tokenizer import TiktokenTokenizer
from src.utils.process_whatsapp import (
    Message,
    clean_messages,
    format_session,
    parse_chat_file,
    segment_conversations,
)

# ──────────────────────────────────────────────────────────────────────────────
# Session → token chunks
# ──────────────────────────────────────────────────────────────────────────────

def _session_to_chunks(
    session: list[Message],
    tokenizer: TiktokenTokenizer,
    sequence_length: int,
) -> list[list[int]]:
    """
    Tokenize a session and split it into chunks of at most *sequence_length*
    tokens, respecting message boundaries so no chunk cuts a message in half.

    An EOS token is appended at the end of each chunk.
    """
    eos_id = tokenizer.get_end_token()

    # Tokenize each message individually and record cumulative lengths.
    msg_tokens: list[list[int]] = []
    for msg in session:
        line = f"{msg.sender}: {msg.text}\n"
        msg_tokens.append(tokenizer.encode(line))

    chunks: list[list[int]] = []
    current: list[int] = []

    for tokens in msg_tokens:
        # If a single message is already too long, hard-truncate it.
        # It's a temporary fix.
        if len(tokens) + 1 > sequence_length:  # +1 for EOS
            tokens = tokens[: sequence_length - 1]

        # If adding this message would overflow the current chunk, flush first.
        if current and len(current) + len(tokens) + 1 > sequence_length:
            chunks.append(current + [eos_id])
            current = []

        current.extend(tokens)

    if current:
        chunks.append(current + [eos_id])

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_finetune(
    input_path: Path,
    output_dir: Path,
    session_gap_minutes: int,
    val_split: float,
    seed: int,
    encoding: str,
    sequence_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TiktokenTokenizer(encoding)
    eos_id = tokenizer.get_end_token()

    # ── Parse, clean, segment ────────────────────────────────────────────────
    print(f"Parsing {input_path} …")
    messages = parse_chat_file(input_path)
    messages = clean_messages(messages)
    sessions = segment_conversations(messages, session_gap_minutes)
    print(f"  {len(messages):,} messages → {len(sessions):,} sessions "
          f"(gap = {session_gap_minutes} min)")

    # ── Tokenize each session into chunks ────────────────────────────────────
    all_chunks: list[list[int]] = []
    for session in sessions:
        all_chunks.extend(_session_to_chunks(session, tokenizer, sequence_length))

    print(f"  {len(all_chunks):,} token chunks after splitting at message boundaries")

    # ── Shuffle and split ────────────────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_chunks)

    n_val = max(1, int(len(all_chunks) * val_split))
    val_chunks   = all_chunks[:n_val]
    train_chunks = all_chunks[n_val:]

    # ── Write binary files ───────────────────────────────────────────────────
    def _write_split(chunks: list[list[int]], name: str) -> dict:
        if not chunks:
            raise ValueError(f"Split '{name}' has no samples.")

        flat    = np.concatenate([np.array(c, dtype=np.uint32) for c in chunks])
        offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
        for i, c in enumerate(chunks):
            offsets[i + 1] = offsets[i] + len(c)

        flat.tofile(output_dir / f"{name}.bin")
        np.save(output_dir / f"{name}_offsets.npy", offsets)

        lengths = [len(c) for c in chunks]
        return {
            "n_samples":  len(chunks),
            "n_tokens":   int(flat.size),
            "min_length": int(min(lengths)),
            "max_length": int(max(lengths)),
            "mean_length": round(float(np.mean(lengths)), 1),
        }

    train_stats = _write_split(train_chunks, "train")
    val_stats   = _write_split(val_chunks,   "val")

    metadata = {
        "encoding":             encoding,
        "eos_token_id":         eos_id,
        "session_gap_minutes":  session_gap_minutes,
        "sequence_length":      sequence_length,
        "val_split":            val_split,
        "seed":                 seed,
        "train":                train_stats,
        "val":                  val_stats,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nOutput → {output_dir}/")
    print(f"  train : {train_stats['n_samples']:>5} samples  "
          f"{train_stats['n_tokens']:>8,} tokens  "
          f"len {train_stats['min_length']}–{train_stats['max_length']}")
    print(f"  val   : {val_stats['n_samples']:>5} samples  "
          f"{val_stats['n_tokens']:>8,} tokens  "
          f"len {val_stats['min_length']}–{val_stats['max_length']}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize a WhatsApp chat export for finetuning."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="WhatsApp .txt export file.")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/finetune"),
                        help="Directory where output files are written (default: ./data/finetune).")
    parser.add_argument("--session-gap", type=int, default=30,
                        help="Minutes of silence that start a new session (default: 30).")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of chunks assigned to validation (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42).")
    parser.add_argument("--encoding", type=str, default="cl100k_base",
                        help="Tiktoken encoding name (default: cl100k_base).")
    parser.add_argument("--sequence-length", type=int, default=2048,
                        help="Maximum tokens per chunk; longer sessions are split "
                             "at message boundaries (default: 2048).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tokenize_finetune(
        input_path=args.input,
        output_dir=args.output_dir,
        session_gap_minutes=args.session_gap,
        val_split=args.val_split,
        seed=args.seed,
        encoding=args.encoding,
        sequence_length=args.sequence_length,
    )
