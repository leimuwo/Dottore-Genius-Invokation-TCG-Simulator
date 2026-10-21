"""Small, explicit PPO learner for variable-cardinality CardEnv batches."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torch import Tensor

from .policy import PolicyValueNet

__all__ = ["PPOConfig", "PPOItem", "PPOTrainer"]


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class PPOItem:
    enc: Tensor
    seat: Tensor
    stage: Tensor
    candidate_features: Tensor
    candidate_kinds: Tensor
    action_index: int
    old_log_prob: float
    advantage: float
    return_: float


class PPOTrainer:
    def __init__(self, model: PolicyValueNet, config: PPOConfig = PPOConfig(),
                 *, optimizer=None):
        if config.learning_rate <= 0 or not 0 < config.clip_range < 1:
            raise ValueError("invalid PPO learning configuration")
        if config.epochs <= 0 or config.minibatch_size <= 0:
            raise ValueError("epochs and minibatch_size must be positive")
        self.model = model
        self.config = config
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=config.learning_rate)

    def update(self, batch: list[PPOItem], *, seed: int = 0) -> dict[str, float]:
        if not batch:
            raise ValueError("cannot update from empty PPO batch")
        if any(not math.isfinite(x.old_log_prob) or not math.isfinite(x.advantage)
               or not math.isfinite(x.return_) for x in batch):
            raise ValueError("PPO batch contains non-finite scalar")
        rng = random.Random(seed)
        indices = list(range(len(batch)))
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "clip_fraction": 0.0, "grad_norm": 0.0, "updates": 0.0}
        for _ in range(self.config.epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                items = [batch[i] for i in indices[start:start + self.config.minibatch_size]]
                losses = []
                clip_count = 0
                for item in items:
                    logits, value = self.model(
                        item.enc, item.seat, item.stage,
                        item.candidate_features, item.candidate_kinds)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = torch.tensor(item.action_index, device=logits.device)
                    log_prob = dist.log_prob(action)
                    ratio = torch.exp(log_prob - item.old_log_prob)
                    advantage = torch.tensor(item.advantage, device=logits.device)
                    return_ = torch.tensor(item.return_, device=logits.device)
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1 - self.config.clip_range,
                                          1 + self.config.clip_range) * advantage
                    policy_loss = -torch.minimum(unclipped, clipped)
                    value_loss = 0.5 * (value - return_).square()
                    entropy = dist.entropy()
                    losses.append(policy_loss + self.config.value_coef * value_loss
                                  - self.config.entropy_coef * entropy)
                    clip_count += int(float(torch.abs(ratio - 1.0).detach()) > self.config.clip_range)
                    metrics["policy_loss"] += float(policy_loss.detach())
                    metrics["value_loss"] += float(value_loss.detach())
                    metrics["entropy"] += float(entropy.detach())
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("non-finite PPO gradient norm")
                self.optimizer.step()
                metrics["grad_norm"] += float(grad_norm)
                metrics["clip_fraction"] += clip_count / len(items)
                metrics["updates"] += 1
        count = len(batch) * self.config.epochs
        for key in ("policy_loss", "value_loss", "entropy", "clip_fraction", "grad_norm"):
            metrics[key] /= count
        return metrics
