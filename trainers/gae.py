"""Generalized Advantage Estimation for one seat's valid trajectory."""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["GAEResult", "compute_gae"]


@dataclass(frozen=True)
class GAEResult:
    advantages: torch.Tensor
    returns: torch.Tensor


def compute_gae(rewards, values, dones, *, gamma: float = 0.99,
                gae_lambda: float = 0.95, next_value: float = 0.0,
                normalize: bool = True) -> GAEResult:
    """Compute GAE without using an opponent's reward as a negation.

    Inputs are ordered transitions for one seat.  ``dones`` marks terminal
    transitions; terminal states never bootstrap.  ``next_value`` is used only
    after a non-terminal final transition.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    r = torch.as_tensor(rewards, dtype=torch.float32)
    v = torch.as_tensor(values, dtype=torch.float32)
    d = torch.as_tensor(dones, dtype=torch.float32)
    if r.ndim != 1 or v.shape != r.shape or d.shape != r.shape:
        raise ValueError("rewards, values and dones must be equal-length vectors")
    if not torch.isfinite(r).all() or not torch.isfinite(v).all():
        raise ValueError("rewards and values must be finite")
    advantages = torch.zeros_like(r)
    gae = torch.tensor(0.0)
    bootstrap = torch.tensor(float(next_value))
    for index in range(len(r) - 1, -1, -1):
        following = torch.zeros((), dtype=torch.float32) if d[index] else (
            bootstrap if index == len(r) - 1 else v[index + 1])
        delta = r[index] + gamma * following - v[index]
        gae = delta + gamma * gae_lambda * (1.0 - d[index]) * gae
        advantages[index] = gae
    returns = advantages + v
    if normalize and len(advantages) > 1:
        std = advantages.std(unbiased=False)
        if float(std) > 1e-8:
            advantages = (advantages - advantages.mean()) / std
    return GAEResult(advantages=advantages, returns=returns)
