"""Minimal M2 training loop joining CardEnv, GAE, PPO, and checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import torch
from .evaluate import load_decks

from dgisim_ext.env_api import CardEnv
from .gae import compute_gae
from .ppo import PPOItem, PPOTrainer
from .policy import PolicyValueNet
from .rollout import Episode, collect_episode, model_decision

__all__ = ["TrainResult", "train_steps"]


def _parallel_episode_worker(model_state, hidden_dim, seed, deck_p1, deck_p2,
                             max_steps, shaping_lambda):
    """Process worker: owns its CardEnv and CPU policy copy."""
    model = PolicyValueNet(hidden_dim=hidden_dim)
    model.load_state_dict(model_state)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    policy = model_decision(model, generator=generator, device="cpu")
    env = CardEnv(shaping_lambda=shaping_lambda)
    return collect_episode(env, seed, policy, max_steps=max_steps,
                           allow_truncated=True, deck_p1=deck_p1, deck_p2=deck_p2)


@dataclass(frozen=True)
class TrainResult:
    choice_steps: int
    valid_episodes: int
    failed_episodes: int
    updates: int
    metrics: dict[str, object]
    timings: dict[str, float]
    batch_sizes: tuple[int, ...]


def _items_from_episode(episode: Episode, gamma: float,
                        gae_lambda: float, model=None) -> list[PPOItem]:
    if not episode.valid or not episode.transitions:
        return []
    # Choice layers are the optimization steps. Rewards are zero at layers
    # before a commit; this preserves the environment's explicit hierarchy.
    by_seat: dict[object, list] = {}
    for transition in episode.transitions:
        by_seat.setdefault(transition.seat, []).append(transition)
    result: list[PPOItem] = []
    for transitions in by_seat.values():
        terminal = bool(episode.transitions[-1].done)
        terminal_rewards = episode.transitions[-1].rewards if terminal else None
        rewards = [x.rewards[x.seat] for x in transitions]
        values = [x.value for x in transitions]
        dones = [x.done for x in transitions]
        # A terminal engine step is observed by the acting seat only.  The
        # opponent's trajectory must nevertheless end at the same payoff;
        # otherwise that seat receives no learning signal for the game result.
        if terminal and terminal_rewards is not None:
            rewards[-1] = terminal_rewards[transitions[-1].seat]
            dones[-1] = True
        next_value = 0.0
        if episode.bootstrap_obs is not None and transitions[-1].seat is episode.bootstrap_obs.seat:
            if model is None:
                raise ValueError("model is required to bootstrap truncated rollout")
            with torch.no_grad():
                next_value = float(model.distribution(episode.bootstrap_obs, device=next(model.parameters()).device)[2])
        gae = compute_gae(
            rewards, values, dones, gamma=gamma,
            gae_lambda=gae_lambda, next_value=next_value, normalize=False)
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
                max_episode_steps: int = 1000,
                training_decks=None, env_workers: int = 1,
                checkpoint_callback=None) -> TrainResult:
    """Run PPO updates on fixed-size rollout batches."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if rollout_steps <= 0 or max_episode_steps <= 0:
        raise ValueError("rollout_steps and max_episode_steps must be positive")
    if env_workers <= 0:
        raise ValueError("env_workers must be positive")
    if training_decks is None:
        training_decks = load_decks("decks/train")
    if len(training_decks) < 2:
        raise ValueError("training_decks must contain at least two decks")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    policy = model_decision(model, generator=generator, device=device)
    count = 0
    game_seed = seed
    env = env_factory()
    shaping_lambda = getattr(env, "_shaping_lambda", 1e-2)
    pending: list[PPOItem] = []
    valid_episodes = failed_episodes = updates = 0
    metric_totals: dict[str, float] = {}
    action_counts = {"swap": 0, "skill": 0, "commit": 0}
    terminal_counts = {"kill": 0, "timeout": 0, "truncated": 0}
    episode_lengths: list[int] = []
    episode_commits: list[int] = []
    truncated_summaries: list[dict] = []
    truncated_tail_actions: list[tuple[dict, ...]] = []
    commit_diagnostics: list[dict] = []
    batch_sizes: list[int] = []
    timings = {"sampling": 0.0, "gae": 0.0, "ppo_update": 0.0}
    pool = None
    if env_workers > 1:
        pool = ProcessPoolExecutor(max_workers=env_workers)

    def flush_batch() -> None:
        nonlocal pending, updates
        if not pending:
            return
        batch_sizes.append(len(pending))
        advantages = torch.tensor([item.advantage for item in pending], dtype=torch.float32)
        if len(advantages) > 1:
            std = advantages.std(unbiased=False)
            if float(std) > 1e-8:
                mean = float(advantages.mean())
                scale = float(std)
                pending = [PPOItem(
                    enc=item.enc, seat=item.seat, stage=item.stage,
                    candidate_features=item.candidate_features,
                    candidate_kinds=item.candidate_kinds,
                    action_index=item.action_index,
                    old_log_prob=item.old_log_prob,
                    advantage=(item.advantage - mean) / scale,
                    return_=item.return_,
                ) for item in pending]
        start = time.perf_counter()
        metrics = trainer.update(pending, seed=seed + updates)
        timings["ppo_update"] += time.perf_counter() - start
        updates += 1
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value
        pending = []

    try:
      while count < steps:
        start = time.perf_counter()
        batch_n = min(env_workers, steps - count)
        jobs = []
        for offset in range(batch_n):
            current_seed = game_seed + offset
            deck_p1 = training_decks[current_seed % len(training_decks)][1]
            deck_p2 = training_decks[(current_seed // len(training_decks) + 1) % len(training_decks)][1]
            max_steps = min(max_episode_steps, steps - count - offset)
            if max_steps <= 0:
                continue
            if pool is None:
                jobs.append(collect_episode(
                    env, current_seed, policy, max_steps=max_steps,
                    allow_truncated=True, deck_p1=deck_p1, deck_p2=deck_p2))
            else:
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                jobs.append(pool.submit(_parallel_episode_worker, state, model.hidden_dim,
                                        current_seed, deck_p1, deck_p2, max_steps,
                                        shaping_lambda))
        episodes = [job.result() if hasattr(job, "result") else job for job in jobs]
        remaining = steps - count
        for episode in episodes:
            if len(episode.transitions) > remaining:
                episode.transitions = episode.transitions[:remaining]
            if not episode.transitions:
                continue
            count += len(episode.transitions)
            remaining = steps - count
            if episode.valid:
                valid_episodes += 1
                episode_lengths.append(len(episode.transitions))
                episode_commits.append(sum(t.action_label == "COMMIT" for t in episode.transitions))
                for transition in episode.transitions:
                    label = transition.action_label
                    if label == "SWAP_CHARACTER": action_counts["swap"] += 1
                    elif label == "CAST_SKILL": action_counts["skill"] += 1
                    if label == "COMMIT": action_counts["commit"] += 1
                if episode.terminal_reason == "timeout": terminal_counts["timeout"] += 1
                elif episode.terminal_reason and episode.terminal_reason.startswith("kill:"):
                    terminal_counts["kill"] += 1
                elif episode.terminal_reason is None:
                    terminal_counts["truncated"] += 1
                    if episode.final_summary is not None:
                        truncated_summaries.append(episode.final_summary)
                    truncated_tail_actions.append(episode.tail_actions)
                start_gae = time.perf_counter()
                pending.extend(_items_from_episode(episode, gamma, gae_lambda, model))
                commit_diagnostics.extend(episode.commit_diagnostics)
                timings["gae"] += time.perf_counter() - start_gae
            else:
                failed_episodes += 1
            if len(pending) >= rollout_steps:
                flush_batch()
            if checkpoint_callback is not None:
                checkpoint_callback(count, model, trainer)
        game_seed += len(episodes)
        timings["sampling"] += time.perf_counter() - start
    finally:
      if pool is not None:
        pool.shutdown()
    flush_batch()
    metrics = ({key: value / updates for key, value in metric_totals.items()}
               if updates else {})
    total_transitions = max(1, sum(action_counts.values()))
    metrics.update({
        "swap_rate": action_counts["swap"] / total_transitions,
        "skill_rate": action_counts["skill"] / total_transitions,
        "commit_rate": action_counts["commit"] / total_transitions,
        "timeout_rate": terminal_counts["timeout"] / max(1, valid_episodes),
        "engine_timeout_rate": terminal_counts["timeout"] / max(1, valid_episodes),
        "kill_rate": terminal_counts["kill"] / max(1, valid_episodes),
        "truncated_rate": terminal_counts["truncated"] / max(1, valid_episodes),
        "episode_length_mean": (sum(episode_lengths) / len(episode_lengths)
                                if episode_lengths else 0.0),
        "commit_count_mean": (sum(episode_commits) / len(episode_commits)
                              if episode_commits else 0.0),
    })
    metrics["truncated_final_summary_count"] = float(len(truncated_summaries))
    metrics["truncated_final_alive_mean"] = (
        sum(s["P1"]["alive"] + s["P2"]["alive"] for s in truncated_summaries)
        / len(truncated_summaries) if truncated_summaries else 0.0)
    metrics["truncated_final_hp_mean"] = (
        sum(s["P1"]["hp"] + s["P2"]["hp"] for s in truncated_summaries)
        / len(truncated_summaries) if truncated_summaries else 0.0)
    # Kept in the manifest as structured diagnostics for the truncation study.
    metrics["truncated_tail_actions"] = truncated_tail_actions
    metrics["commit_diagnostics"] = commit_diagnostics
    return TrainResult(count, valid_episodes, failed_episodes, updates, metrics,
                       timings, tuple(batch_sizes))
