from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int
from torch import Tensor

from transformer_module import softmax as module_softmax


def softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    return module_softmax(in_features, dim=dim)


def cross_entropy(
    inputs: Float[Tensor, "batch vocab"],
    targets: Int[Tensor, " batch"],
) -> Float[Tensor, ""]:
    # Stable log-softmax CE; subtract max, avoid materializing a huge exp buffer twice.
    x_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - x_max
    log_sum_exp = torch.logsumexp(shifted, dim=-1) + x_max.squeeze(-1)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.detach().float().pow(2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef.to(g.dtype))


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])
    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters if warmup_iters > 0 else max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]
                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay (handout / AdamW).
                p.mul_(1 - lr * weight_decay)

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)

        return loss


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["iteration"])
