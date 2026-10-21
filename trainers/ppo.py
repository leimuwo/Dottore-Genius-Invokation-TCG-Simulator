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
                enc, seat, stage, features, kinds, mask, actions, old_log_probs, advantages, returns = self._collate(items)
                logits, values = self.model(enc, seat, stage, features, kinds)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                    values = values.unsqueeze(0)
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(log_prob - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1 - self.config.clip_range,
                                      1 + self.config.clip_range) * advantages
                policy_loss = -torch.minimum(unclipped, clipped)
                value_loss = 0.5 * (values - returns).square()
                entropy = -(probs * log_probs).masked_fill(~mask, 0.0).sum(dim=-1)
                loss = (policy_loss + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy).mean()
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
                metrics["policy_loss"] += float(policy_loss.detach().mean())
                metrics["value_loss"] += float(value_loss.detach().mean())
                metrics["entropy"] += float(entropy.detach().mean())
                metrics["clip_fraction"] += float((torch.abs(ratio - 1.0) > self.config.clip_range).float().mean())
                metrics["updates"] += 1
        count = metrics["updates"]
        for key in ("policy_loss", "value_loss", "entropy", "clip_fraction", "grad_norm"):
            metrics[key] /= count
        return metrics

    def _collate(self, items):
        batch = len(items)
        max_choices = max(item.candidate_features.shape[0] for item in items)
        feature_dim = self.model.feature_dim
        enc = torch.stack([item.enc for item in items])
        seat = torch.stack([item.seat for item in items])
        stage = torch.stack([item.stage for item in items])
        features = torch.zeros(batch, max_choices, feature_dim)
        kinds = torch.zeros(batch, max_choices, dtype=torch.long)
        mask = torch.zeros(batch, max_choices, dtype=torch.bool)
        for row, item in enumerate(items):
            n = item.candidate_features.shape[0]
            width = item.candidate_features.shape[-1]
            if width > feature_dim:
                raise ValueError("PPO item feature width exceeds model feature_dim")
            features[row, :n, :width] = item.candidate_features
            kinds[row, :n] = item.candidate_kinds
            mask[row, :n] = True
        actions = torch.tensor([item.action_index for item in items], dtype=torch.long)
        old_log_probs = torch.tensor([item.old_log_prob for item in items])
        advantages = torch.tensor([item.advantage for item in items])
        returns = torch.tensor([item.return_ for item in items])
        return enc, seat, stage, features, kinds, mask, actions, old_log_probs, advantages, returns
