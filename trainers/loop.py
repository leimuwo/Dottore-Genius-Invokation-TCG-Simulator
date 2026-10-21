"""Minimal M2 training loop joining CardEnv, GAE, PPO, and checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from dgisim_ext.env_api import CardEnv
from .gae import compute_gae
from .ppo import PPOItem, PPOTrainer
from .rollout import Episode, collect_episode, model_decision

__all__ = ["TrainResult", "train_steps"]


@dataclass(frozen=True)
class TrainResult:
    choice_steps: int
    valid_episodes: int
    failed_episodes: int
    updates: int
    metrics: dict[str, float]


def _items_from_episode(episode: Episode, gamma: float,
                        gae_lambda: float) -> list[PPOItem]:
    if not episode.valid or not episode.transitions:
        return []
    # Choice layers are the optimization steps. Rewards are zero at layers
    # before a commit; this preserves the environment's explicit hierarchy.
    by_seat: dict[object, list] = {}
    for transition in episode.transitions:
        by_seat.setdefault(transition.seat, []).append(transition)
    result: list[PPOItem] = []
    for transitions in by_seat.values():
        gae = compute_gae(
            [x.rewards[x.seat] for x in transitions], [x.value for x in transitions],
            [x.done for x in transitions], gamma=gamma,
            gae_lambda=gae_lambda, normalize=True)
        for transition, advantage, return_ in zip(
                transitions, gae.advantages, gae.returns):
            result.append(PPOItem(
                enc=torch.tensor(transition.enc, dtype=torch.float32),
                seat=torch.tensor(float(transition.seat.value)),
                stage=torch.tensor(float(transition.stage_index)),
                candidate_features=torch.tensor(transition.candidate_features, dtype=torch.float32),
                candidate_kinds=torch.tensor([
                    {"tuple": 0, "card": 1, "dice": 2, "cost": 3,
                     "commit": 4, "card_count": 5, "dice_count": 6,
                     "cost_count": 7}[kind]
                    for kind in transition.candidate_kinds]),
                action_index=transition.action_index,
                old_log_prob=transition.log_prob,
                advantage=float(advantage), return_=float(return_),
            ))
    return result


def train_steps(model, trainer: PPOTrainer, *, steps: int = 1000,
                seed: int = 0, gamma: float = 0.99,
                gae_lambda: float = 0.95,
                env_factory: Callable[[], CardEnv] = CardEnv) -> TrainResult:
    """Run a small reproducible current-policy self-play loop."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    generator = torch.Generator().manual_seed(seed)
    policy = model_decision(model, generator=generator)
    episodes: list[Episode] = []
    count = 0
    game_seed = seed
    env = env_factory()
    while count < steps:
        episode = collect_episode(
            env, game_seed, policy,
            max_steps=min(1000, steps - count), allow_truncated=True)
        episodes.append(episode)
        remaining = steps - count
        if len(episode.transitions) > remaining:
            episode.transitions = episode.transitions[:remaining]
        count += len(episode.transitions)
        game_seed += 1
    batch = []
    for episode in episodes:
        batch.extend(_items_from_episode(episode, gamma, gae_lambda))
    metrics = trainer.update(batch, seed=seed)
    return TrainResult(count, sum(e.valid for e in episodes),
                       sum(not e.valid for e in episodes), 1, metrics)
