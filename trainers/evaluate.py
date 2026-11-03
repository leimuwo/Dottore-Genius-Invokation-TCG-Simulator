"""Fixed-deck evaluation with seat-stratified Wilson confidence intervals."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from dataclasses import dataclass

from dgisim_ext.env_api import CardEnv
from src.dgisim.deck import FrozenDeck
from src.dgisim.state.enums import Pid
from .rollout import Decision, random_decision

__all__ = ["wilson_interval", "load_decks", "attack_first_decision", "evaluate_policy"]


def attack_first_decision(obs) -> Decision:
    """Conservative attack-first baseline without swap loops.

    A swap is never selected merely because it is available.  The public
    observation does not expose the post-swap candidate tree, so accepting a
    swap without proving that a follow-up skill is payable would make the
    baseline manufacture long swap cycles.  Skills are selected first; when
    none is currently legal, ending the round is the safe fallback.
    """
    skill_indices = [i for i, choice in enumerate(obs.choices)
                     if choice.action_type == "skill"]
    if skill_indices:
        index = skill_indices[0]
    else:
        end_indices = [i for i, choice in enumerate(obs.choices)
                       if choice.action_type == "end_round"]
        index = end_indices[0] if end_indices else 0
    count = len(obs.choices)
    return Decision(index, -math.log(max(1, count)), 0.0, 0.0)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return center - radius, center + radius


def load_decks(folder: str | Path) -> list[tuple[str, FrozenDeck]]:
    rows = []
    for path in sorted(Path(folder).glob("*.json")):
        deck = FrozenDeck.from_json(path.read_text(encoding="utf-8"))
        if deck is None:
            raise ValueError(f"cannot load deck: {path}")
        rows.append((path.stem, deck))
    return rows


@dataclass(frozen=True)
class EvalResult:
    games: int
    wins: int
    kills: int
    timeouts: int
    payoff_p1: float
    payoff_p2: float
    winrate_ci95: tuple[float, float]
    by_seat: dict[str, dict]


def evaluate_policy(model_policy, random_policy, *, deck_folder="decks/test",
                    seeds=(0,), model_seat: Pid = Pid.P1,
                    shaping: bool = False) -> EvalResult:
    decks = load_decks(deck_folder)
    if not decks:
        raise ValueError("no evaluation decks found")
    wins = kills = timeouts = 0
    payoff = {Pid.P1: 0.0, Pid.P2: 0.0}
    by_seat = {"P1": {"wins": 0, "games": 0, "kills": 0,
                       "timeouts": 0, "payoff": 0.0},
               "P2": {"wins": 0, "games": 0, "kills": 0,
                       "timeouts": 0, "payoff": 0.0}}
    games = 0
    for left_name, left in decks:
        for right_name, right in decks:
            for seed in seeds:
                env = CardEnv(shaping=shaping)
                rng = random.Random(seed + 1000003)
                if random_policy is None:
                    opponent_choice = random_decision(rng)
                elif callable(random_policy):
                    opponent_choice = random_policy
                else:
                    opponent_choice = random_policy
                obs = env.reset(left, right, seed)
                while True:
                    policy = model_policy if obs.seat is model_seat else opponent_choice
                    decision = policy(obs)
                    out = env.step(decision.action_index)
                    if out.done:
                        break
                    obs = out.obs
                games += 1
                rewards = out.info["rewards"]
                payoff[Pid.P1] += rewards[Pid.P1]
                payoff[Pid.P2] += rewards[Pid.P2]
                reason = out.terminal_reason or ""
                if reason == "timeout":
                    timeouts += 1
                elif reason.startswith("kill:"):
                    kills += 1
                seat_row = by_seat[model_seat.name]
                seat_row.setdefault("payoff", 0.0)
                seat_row["payoff"] += rewards[model_seat]
                seat_row.setdefault("kills", 0)
                seat_row.setdefault("timeouts", 0)
                seat_row["kills" if reason.startswith("kill:") else "timeouts"] += 1
                if rewards[model_seat] > 0:
                    wins += 1
                    seat_row["wins"] += 1
                seat_row["games"] += 1
    lo, hi = wilson_interval(wins, games)
    return EvalResult(games, wins, kills, timeouts, payoff[Pid.P1], payoff[Pid.P2],
                      (lo, hi), by_seat)
