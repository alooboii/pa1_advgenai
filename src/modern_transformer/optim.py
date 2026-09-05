from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean next-token cross entropy without torch.nn.functional.cross_entropy."""
    # BEGIN SOLUTION
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match every logits dimension except vocabulary")
    maxima = logits.amax(dim=-1, keepdim=True)
    shifted = logits - maxima
    log_partition = shifted.exp().sum(dim=-1).log() + maxima.squeeze(-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_partition - target_logits).mean()
    # END SOLUTION


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps, and weight_decay must be non-negative")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must be in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        # BEGIN SOLUTION
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)
        return loss
        # END SOLUTION
