"""Minimal M2 training loop joining CardEnv, GAE, PPO, and checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import time
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
    timings: dict[str, float]
    batch_sizes: tuple[int, ...]


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
                env_factory: Callable[[], CardEnv] = CardEnv,
                rollout_steps: int = 2048,
                max_episode_steps: int = 1000) -> TrainResult:
    """Run PPO updates on fixed-size rollout batches."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if rollout_steps <= 0 or max_episode_steps <= 0:
        raise ValueError("rollout_steps and max_episode_steps must be positive")
    generator = torch.Generator().manual_seed(seed)
    policy = model_decision(model, generator=generator)
    count = 0
    game_seed = seed
    env = env_factory()
    pending: list[PPOItem] = []
    valid_episodes = failed_episodes = updates = 0
    metric_totals: dict[str, float] = {}
    batch_sizes: list[int] = []
    timings = {"sampling": 0.0, "gae": 0.0, "ppo_update": 0.0}

    def flush_batch() -> None:
        nonlocal pending, updates
        if not pending:
            return
        batch_sizes.append(len(pending))
        start = time.perf_counter()
        metrics = trainer.update(pending, seed=seed + updates)
        timings["ppo_update"] += time.perf_counter() - start
        updates += 1
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value
        pending = []

    while count < steps:
        start = time.perf_counter()
        episode = collect_episode(
            env, game_seed, policy,
            max_steps=min(max_episode_steps, steps - count), allow_truncated=True)
        remaining = steps - count
        if len(episode.transitions) > remaining:
            episode.transitions = episode.transitions[:remaining]
        count += len(episode.transitions)
        timings["sampling"] += time.perf_counter() - start
        if episode.valid:
            valid_episodes += 1
            start = time.perf_counter()
            pending.extend(_items_from_episode(episode, gamma, gae_lambda))
            timings["gae"] += time.perf_counter() - start
        else:
            failed_episodes += 1
        if len(pending) >= rollout_steps:
            flush_batch()
        game_seed += 1
        if not episode.transitions:
            raise RuntimeError("rollout made no progress")
    flush_batch()
    metrics = ({key: value / updates for key, value in metric_totals.items()}
               if updates else {})
    return TrainResult(count, valid_episodes, failed_episodes, updates, metrics,
                       timings, tuple(batch_sizes))
