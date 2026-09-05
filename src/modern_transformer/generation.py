from __future__ import annotations

import torch
from torch import Tensor

from .model import TransformerLM


@torch.inference_mode()
def generate(
    model: TransformerLM,
    token_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    # BEGIN SOLUTION
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    output = token_ids
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = output[:, -model.config.max_seq_len :]
        logits = model(context)[:, -1]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                threshold = logits.topk(k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
        output = torch.cat((output, next_token), dim=1)
        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break
    model.train(was_training)
    return output
    # END SOLUTION
