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

__all__ = ["wilson_interval", "evaluate_policy"]


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
                    seeds=(0,), model_seat: Pid = Pid.P1) -> EvalResult:
    decks = load_decks(deck_folder)
    if not decks:
        raise ValueError("no evaluation decks found")
    wins = kills = timeouts = 0
    payoff = {Pid.P1: 0.0, Pid.P2: 0.0}
    by_seat = {"P1": [0, 0], "P2": [0, 0]}  # wins, games
    games = 0
    for left_name, left in decks:
        for right_name, right in decks:
            for seed in seeds:
                env = CardEnv()
                rng = random.Random(seed + 1000003)
                random_choice = random_policy or random_decision(rng)
                obs = env.reset(left, right, seed)
                while True:
                    policy = model_policy if obs.seat is model_seat else random_choice
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
                else:
                    kills += 1
                if rewards[model_seat] > 0:
                    wins += 1
                    by_seat[model_seat.name][0] += 1
                by_seat[model_seat.name][1] += 1
    lo, hi = wilson_interval(wins, games)
    return EvalResult(games, wins, kills, timeouts, payoff[Pid.P1], payoff[Pid.P2],
                      (lo, hi), {k: {"wins": v[0], "games": v[1]} for k, v in by_seat.items()})
