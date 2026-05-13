"""
Autoregressive inference for the friendsbot language model.

Loads a checkpoint, reads a prompt from stdin, and streams generated tokens
to stdout one by one.  Generation stops when the end-of-text token is
produced or --max-new-tokens is reached.

Sampling pipeline applied to each step's logits:
    repetition penalty → temperature → top-k → top-p → softmax → sample

Usage:
    python -m src.generate \\
        --checkpoint checkpoints/checkpoint_last.pt \\
        --config configs/training.yaml \\
        --max-new-tokens 200 \\
        --temperature 0.8 \\
        --top-k 50 \\
        --top-p 0.95 \\
        --repetition-penalty 1.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.tokenizer import TiktokenTokenizer
from src.train import load_training_config, resolve_device
from src.transformer import LanguageModel, ModelConfig, get_supported_weights_precision


def _load_model(
    checkpoint_path: Path,
    model_config: ModelConfig,
    device: torch.device,
) -> LanguageModel:
    model = LanguageModel.from_config(model_config, kv_cache={}, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _reset_state(model: LanguageModel) -> None:
    """Clear the KV-cache and token counter before each generation."""
    model.transformer_decoder.global_token_counter = 0
    model.transformer_decoder.kv_cache.clear()


def _sample_next_token(
    logits: torch.Tensor,
    past_ids: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> int:
    """
    Apply the full sampling pipeline to a (vocab_size,) logit vector and
    return the sampled token id.

    Steps:
        1. Repetition penalty  — down-weight tokens that already appear in past_ids.
        2. Temperature         — sharpen (< 1) or flatten (> 1) the distribution.
        3. Top-k               — zero out all but the k highest-logit tokens.
        4. Top-p               — zero out tokens beyond the nucleus of probability p.
        5. Softmax + sample    — convert to probabilities and draw one token.

    temperature=0 short-circuits to greedy argmax.
    """
    if temperature == 0.0:
        return int(logits.argmax().item())

    logits = logits.clone().float()

    # 1. Repetition penalty: divide positive logits and multiply negative ones
    #    so that already-seen tokens become relatively less likely.
    if repetition_penalty != 1.0 and past_ids:
        for token_id in set(past_ids):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    # 2. Temperature
    logits /= temperature

    # 3. Top-k: keep only the k tokens with the highest logits.
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k).values[-1]
        logits[logits < cutoff] = float("-inf")

    # 4. Top-p (nucleus): starting from the highest-probability token, keep
    #    adding tokens until their cumulative probability reaches p, then
    #    discard everything else.
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens whose inclusion would push the cumulative total past p
        # (shift by one so the token that crosses the threshold is kept).
        sorted_to_remove = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits[sorted_to_remove] = float("-inf")
        logits.scatter_(0, sorted_indices, sorted_logits)

    # 5. Softmax + 6. Sample
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def generate(
    model: LanguageModel,
    prompt_ids: list[int],
    eos_id: int,
    tokenizer: TiktokenTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> None:
    _reset_state(model)

    # past_ids tracks every token seen so far (prompt + generated) for the
    # repetition penalty; it lives on CPU as a plain Python list.
    past_ids: list[int] = list(prompt_ids)

    # Prefill: process the full prompt in one forward pass to populate the KV-cache.
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        logits = model(x)  # (1, T_prompt, vocab_size)
    next_token = _sample_next_token(
        logits[0, -1], past_ids, temperature, top_k, top_p, repetition_penalty
    )

    generated: list[int] = []
    prev_text = ""

    while next_token != eos_id and len(generated) < max_new_tokens:
        generated.append(next_token)
        past_ids.append(next_token)

        # Decode the full generated list each time so that multi-byte UTF-8
        # tokens that are only valid in sequence are handled correctly.
        new_text = tokenizer.decode(generated)
        print(new_text[len(prev_text):], end="", flush=True)
        prev_text = new_text

        # Decode step: single token; the KV-cache provides the full context.
        x = torch.tensor([[next_token]], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)  # (1, 1, vocab_size)
        next_token = _sample_next_token(
            logits[0, -1], past_ids, temperature, top_k, top_p, repetition_penalty
        )

    print()  # trailing newline


def main() -> None:
    args = _parse_args()

    config = load_training_config(args.config)
    model_config = ModelConfig.from_yaml(config.model_config)
    device = resolve_device(config.device)
    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type in ("cuda", "mps")
    tokenizer = TiktokenTokenizer(config.tokenizer_encoding)
    eos_id = tokenizer.get_end_token()

    print(f"Loading checkpoint from {args.checkpoint} …", file=sys.stderr)
    model = _load_model(args.checkpoint, model_config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready  ({n_params:,} parameters, device={device})\n", file=sys.stderr)

    prompt = input("Prompt: ")
    if not prompt:
        print("Empty prompt — exiting.", file=sys.stderr)
        return

    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) >= model_config.sequence_length:
        print(
            f"Prompt is {len(prompt_ids)} tokens but the model's sequence length is "
            f"{model_config.sequence_length}. Please use a shorter prompt.",
            file=sys.stderr,
        )
        return

    # Move the cursor back up to the prompt line so generated tokens appear
    # on the same line as the user's input rather than on a new one.
    print(f"\033[1A\rPrompt: {prompt}", end="", flush=True)

    generate(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=eos_id,
        tokenizer=tokenizer,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoregressive inference for the friendsbot language model."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training.yaml"),
        help="Path to the YAML training config (provides model arch and tokenizer).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="Maximum tokens to generate before stopping (default: 200).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature. 0 = greedy, <1 = sharper, >1 = flatter (default: 1.0).",
    )
    parser.add_argument(
        "--top-k", type=int, default=0,
        help="Keep only the top-k tokens before sampling. 0 = disabled (default: 0).",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0,
        help="Nucleus sampling: keep the smallest set of tokens whose cumulative "
             "probability reaches p. 1.0 = disabled (default: 1.0).",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="Penalty applied to logits of tokens already seen. "
             "1.0 = no penalty, >1.0 discourages repetition (default: 1.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
