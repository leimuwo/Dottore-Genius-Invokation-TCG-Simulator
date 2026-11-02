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
                             max_steps, shaping_lambda, shaping):
    """Process worker: owns its CardEnv and CPU policy copy."""
    model = PolicyValueNet(hidden_dim=hidden_dim)
    model.load_state_dict(model_state)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    policy = model_decision(model, generator=generator, device="cpu")
    env = CardEnv(shaping_lambda=shaping_lambda, shaping=shaping)
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
    # Choice layers remain the optimization steps, but rewards are assigned on
    # each seat's decision interval. This includes rewards emitted while the
    # opponent acts and prevents terminal payoff from overwriting shaping.
    by_seat: dict[object, list] = {}
    for transition in episode.transitions:
        by_seat.setdefault(transition.seat, []).append(transition)
    ledger = _decision_interval_rewards(episode, by_seat, gamma=gamma)
    interval_gammas = _decision_interval_gammas(episode, by_seat, gamma)
    result: list[PPOItem] = []
    for seat, transitions in by_seat.items():
        rewards = ledger[seat]
        values = [x.value for x in transitions]
        dones = [x.done for x in transitions]
        # Choice construction is instantaneous; only a committed action
        # advances the game clock and receives the configured discount.
        gammas = interval_gammas[seat]
        if episode.transitions[-1].done:
            # The terminal payoff is present in the final transition's reward
            # map. Both seat trajectories end at the same game terminal.
            dones[-1] = True
        next_value = 0.0
        bootstrap_obs = episode.bootstrap_obs_by_seat.get(seat)
        if bootstrap_obs is not None:
            if model is None:
                raise ValueError("model is required to bootstrap truncated rollout")
            with torch.no_grad():
                next_value = float(model.value_from_obs(
                    bootstrap_obs, device=next(model.parameters()).device))
        gae = compute_gae(
            rewards, values, dones, gamma=gamma, gammas=gammas,
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
                    candidate_action_types=torch.tensor([
                        {"tuple": 0, "skill": 1, "card": 2, "swap": 3,
                         "end_round": 4, "reroll": 5, "tuning": 6,
                         "dice": 7, "cost": 8, "commit": 9}.get(kind, 0)
                        for kind in transition.candidate_action_types]),
                    action_index=transition.action_index,
                    old_log_prob=transition.log_prob,
                    advantage=float(advantage), return_=float(return_),
                ))
    return result


def _decision_interval_rewards(episode: Episode, by_seat: dict, gamma: float = 1.0) -> dict:
    """Accumulate each seat's rewards until its next decision.

    A reward map is emitted after every committed action, including the
    opponent's action. The acting seat starts a new interval before its own
    reward is added; all other seats keep accumulating that reward in their
    current interval. This makes terminal payoff additive and preserves
    opponent/automatic shaping in both trajectories.
    """
    rewards = {seat: [0.0] * len(transitions)
               for seat, transitions in by_seat.items()}
    positions = {seat: index for seat, index in
                 ((seat, 0) for seat in by_seat)}
    pending = {seat: 0.0 for seat in by_seat}
    elapsed = {seat: 0 for seat in by_seat}
    seen = {seat: False for seat in by_seat}
    for transition in episode.transitions:
        seat = transition.seat
        if seen[seat]:
            rewards[seat][positions[seat]] += pending[seat]
            positions[seat] += 1
            pending[seat] = 0.0
            elapsed[seat] = 0
        else:
            seen[seat] = True
        for target, value in transition.rewards.items():
            if target in pending:
                pending[target] += (gamma ** elapsed[target]) * float(value)
        for target in pending:
            if (seen[target]
                    and getattr(transition, "action_label", None) == "COMMIT"):
                elapsed[target] += 1
    for seat in by_seat:
        rewards[seat][positions[seat]] += pending[seat]
    return rewards


def _decision_interval_gammas(episode: Episode, by_seat: dict, gamma: float) -> dict:
    """Discount from each seat decision to its next decision boundary."""
    positions = {seat: [i for i, t in enumerate(episode.transitions)
                         if t.seat is seat] for seat in by_seat}
    result = {}
    for seat, indices in positions.items():
        values = []
        for pos, index in enumerate(indices):
            end = indices[pos + 1] if pos + 1 < len(indices) else len(episode.transitions)
            commits = sum(episode.transitions[j].action_label == "COMMIT"
                          for j in range(index, end))
            values.append(gamma ** commits)
        result[seat] = values
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
    shaping_enabled = getattr(env, "_shaping", True)
    pending: list[PPOItem] = []
    valid_episodes = failed_episodes = updates = 0
    metric_totals: dict[str, float] = {}
    action_counts = {"swap": 0, "skill": 0, "commit": 0}
    commit_action_counts: dict[str, int] = {}
    choice_counts: dict[str, int] = {}
    terminal_counts = {"kill": 0, "timeout": 0, "truncated": 0}
    truncation_counts = {"budget": 0, "max_episode_steps": 0}
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
                    candidate_action_types=item.candidate_action_types,
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
        remaining_budget = steps - count
        caps = [remaining_budget // batch_n
                + (1 if offset < remaining_budget % batch_n else 0)
                for offset in range(batch_n)]
        for offset in range(batch_n):
            current_seed = game_seed + offset
            deck_p1 = training_decks[current_seed % len(training_decks)][1]
            deck_p2 = training_decks[(current_seed // len(training_decks) + 1) % len(training_decks)][1]
            max_steps = min(max_episode_steps, caps[offset])
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
                                        shaping_lambda, shaping_enabled))
        episodes = [job.result() if hasattr(job, "result") else job for job in jobs]
        remaining = steps - count
        for episode_index, episode in enumerate(episodes):
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
                    choice_counts[label] = choice_counts.get(label, 0) + 1
                    if label == "SWAP_CHARACTER": action_counts["swap"] += 1
                    elif label == "CAST_SKILL": action_counts["skill"] += 1
                    if label == "COMMIT": action_counts["commit"] += 1
                if episode.terminal_reason == "timeout": terminal_counts["timeout"] += 1
                elif episode.terminal_reason and episode.terminal_reason.startswith("kill:"):
                    terminal_counts["kill"] += 1
                elif episode.terminal_reason is None:
                    terminal_counts["truncated"] += 1
                    # A worker is capped by both the remaining global budget
                    # and max_episode_steps. Classify the boundary explicitly
                    # so these two causes are never reported as one rate.
                    episode_budget_end = count >= steps
                    cap = caps[episode_index]
                    if episode_budget_end or cap < max_episode_steps:
                        truncation_counts["budget"] += 1
                    else:
                        truncation_counts["max_episode_steps"] += 1
                    if episode.final_summary is not None:
                        truncated_summaries.append(episode.final_summary)
                    truncated_tail_actions.append(episode.tail_actions)
                start_gae = time.perf_counter()
                pending.extend(_items_from_episode(episode, gamma, gae_lambda, model))
                commit_diagnostics.extend(episode.commit_diagnostics)
                for diagnostic in episode.commit_diagnostics:
                    kind = diagnostic["action_type"]
                    commit_action_counts[kind] = commit_action_counts.get(kind, 0) + 1
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
    total_commits = max(1, action_counts["commit"])
    metrics.update({
        # These rates are over committed game actions, not layered choices.
        # A skill/swap may have several preceding payment choices.
        "swap_rate": commit_action_counts.get("swap", 0) / total_commits,
        "skill_rate": commit_action_counts.get("skill", 0) / total_commits,
        "commit_rate": action_counts["commit"] / total_commits,
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
    metrics["truncation_budget_count"] = truncation_counts["budget"]
    metrics["truncation_max_episode_steps_count"] = truncation_counts["max_episode_steps"]
    metrics["truncation_budget_rate"] = truncation_counts["budget"] / max(1, valid_episodes)
    metrics["truncation_max_episode_steps_rate"] = (
        truncation_counts["max_episode_steps"] / max(1, valid_episodes))
    metrics["commit_diagnostics"] = commit_diagnostics
    metrics["choice_counts"] = choice_counts
    metrics["commit_action_counts"] = commit_action_counts
    metrics["choice_count_total"] = sum(choice_counts.values())
    metrics["commit_count_total"] = sum(commit_action_counts.values())
    return TrainResult(count, valid_episodes, failed_episodes, updates, metrics,
                       timings, tuple(batch_sizes))
