import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.tokenizer import TiktokenTokenizer
from src.train import load_training_config, validate_device
from src.transformer import LanguageModel, get_supported_weights_precision


def sample_next_token(
    logits: torch.Tensor,
    past_ids: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> int:
    if temperature == 0.0:
        return int(logits.argmax().item())

    logits = logits.clone().float()

    # Repetition penalty
    if repetition_penalty != 1.0 and past_ids:
        for token_id in set(past_ids):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    # Temperature
    logits /= temperature

    # Top-K
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k).values[-1]
        logits[logits < cutoff] = float("-inf")

    # Nucleulus
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_to_remove = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_to_remove[0] = False
        sorted_logits[sorted_to_remove] = float("-inf")
        logits.scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
    


def chat(config_file: Path,
         checkpoint_path: Path,
         max_new_tokens: int,
         temperature: float,
         top_k: int,
         top_p: float,
         repetition_penalty: float) -> None:
    config = load_training_config(config_file)
    device = validate_device(config["device"])
    amp_dtype = get_supported_weights_precision(device)
    use_amp = device.type in ("cuda", "mps")
    tokenizer = TiktokenTokenizer(config["tokenizer_encoding"])
    eos_id = tokenizer.get_end_token()

    print(f"Loading checkpoint from {checkpoint_path}...")
    model = LanguageModel.from_config(config["model_config"], kv_cache={}, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Used for repetition penalty
    past_ids: list[int]
    generated: list[int]
    prev_text: str

    while True:
        prompt = input("Human: ")
        if not prompt:
            print("Empty prompt — exiting.", file=sys.stderr)
            return

        prompt_ids = tokenizer.encode(f"Human: {prompt}\nModel:")
        if len(prompt_ids) >= model.sequence_length:
            print(
                f"Prompt is {len(prompt_ids)} tokens but the model's sequence length is "
                f"{model.sequence_length}. Please use a shorter prompt.",
                file=sys.stderr,
            )
            return

        with torch.no_grad():
            model.transformer_decoder.global_token_counter = 0
            model.transformer_decoder.kv_cache.clear()
            past_ids = []
            generated = []
            prev_text = ""

            # KV-cache prefill
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)  # (1, T_prompt, vocab_size)
            next_token = sample_next_token(
                logits[0, -1], past_ids, temperature, top_k, top_p, repetition_penalty
            )
                
            print("Model:", end="")
            while next_token != eos_id and len(generated) < max_new_tokens:
                generated.append(next_token)
                past_ids.append(next_token)
                new_text = tokenizer.decode(generated)
                print(new_text[len(prev_text):], end="", flush=True)
                prev_text = new_text

                # Using KV-cache
                x = torch.tensor([[next_token]], dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)  # (1, 1, vocab_size)
                next_token = sample_next_token(
                    logits[0, -1], past_ids, temperature, top_k, top_p, repetition_penalty
                )

            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the finetuned language model."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Maximum tokens to generate before stopping (default: 200).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
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
    args = parse_args()
    chat(args.config,
         args.checkpoint,
         args.max_new_tokens,
         args.temperature,
         args.top_k,
         args.top_p,
         args.repetition_penalty)
