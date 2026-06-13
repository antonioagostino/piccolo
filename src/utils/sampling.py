import torch
import torch.nn.functional as F

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

    # Nucleus
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_to_remove = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_to_remove[0] = False
        sorted_logits[sorted_to_remove] = float("-inf")
        logits.scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())