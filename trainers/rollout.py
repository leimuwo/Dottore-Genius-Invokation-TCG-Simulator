"""M2 rollout collection for the layered CardEnv interface."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from dgisim_ext.env_api import CardEnv, CardEnvError, Obs
from src.dgisim.state.enums import Pid

__all__ = [
    "Decision", "Transition", "Episode", "random_decision",
    "model_decision", "collect_episode", "collect_rollout",
]


@dataclass(frozen=True)
class Decision:
    action_index: int
    log_prob: float
    value: float
    entropy: float


@dataclass(frozen=True)
class Transition:
    obs_fingerprint: str
    seat: Pid
    candidate_features: tuple[tuple[float, ...], ...]
    candidate_kinds: tuple[str, ...]
    action_index: int
    log_prob: float
    value: float
    entropy: float
    reward: float
    rewards: dict[Pid, float]
    done: bool
    terminal_reason: str | None


@dataclass
class Episode:
    seed: int
    transitions: list[Transition] = field(default_factory=list)
    valid: bool = True
    failure: dict | None = None
    terminal_reason: str | None = None


class Policy(Protocol):
    def __call__(self, obs: Obs) -> Decision: ...


def obs_fingerprint(obs: Obs) -> str:
    """Stable fingerprint of public observation and candidate content."""
    payload = repr((obs.seat.name, obs.enc, obs.stage_index,
                    tuple((c.kind, c.feature) for c in obs.choices))).encode()
    return hashlib.sha256(payload).hexdigest()


def random_decision(rng: random.Random) -> Policy:
    """Return a uniform candidate policy with explicit strategy RNG."""
    def choose(obs: Obs) -> Decision:
        count = len(obs.choices)
        if not count:
            raise ValueError("cannot choose from empty candidate set")
        return Decision(rng.randrange(count), -math.log(count), 0.0, math.log(count))
    return choose


def model_decision(model, *, generator: torch.Generator | None = None,
                   deterministic: bool = False, device=None) -> Policy:
    """Adapt ``PolicyValueNet.act`` to the rollout policy protocol."""
    def choose(obs: Obs) -> Decision:
        # ``act`` currently builds tensors on the default device.  Device is
        # kept in the adapter signature so the rollout interface can remain
        # stable when GPU batching is added.
        del device
        index, log_prob, value, entropy = model.act(
            obs, deterministic=deterministic, generator=generator)
        return Decision(index, float(log_prob), float(value), float(entropy))
    return choose


def collect_episode(env: CardEnv, seed: int, policy: Policy,
                    *, max_steps: int = 10000) -> Episode:
    episode = Episode(seed=seed)
    try:
        obs = env.reset(seed=seed)
        for _ in range(max_steps):
            if not obs.choices:
                raise CardEnvError("exception", "observation has no choices")
            decision = policy(obs)
            if not 0 <= decision.action_index < len(obs.choices):
                raise CardEnvError("illegal", "policy returned invalid choice")
            choice = obs.choices[decision.action_index]
            out = env.step(decision.action_index)
            episode.transitions.append(Transition(
                obs_fingerprint=obs_fingerprint(obs),
                seat=obs.seat,
                candidate_features=tuple(c.feature for c in obs.choices),
                candidate_kinds=tuple(c.kind for c in obs.choices),
                action_index=decision.action_index,
                log_prob=decision.log_prob,
                value=decision.value,
                entropy=decision.entropy,
                reward=out.reward,
                rewards=dict(out.info.get("rewards", {})),
                done=out.done,
                terminal_reason=out.terminal_reason,
            ))
            if out.done:
                episode.terminal_reason = out.terminal_reason
                return episode
            if out.obs is None:
                raise CardEnvError("exception", "non-terminal step returned no obs")
            obs = out.obs
        raise CardEnvError("noop", f"rollout exceeded {max_steps} steps")
    except CardEnvError as exc:
        episode.valid = False
        episode.failure = {"category": exc.category, "message": str(exc)}
        return episode
    except Exception as exc:  # diagnostics must not enter a PPO batch
        episode.valid = False
        episode.failure = {"category": "exception", "message": f"{type(exc).__name__}: {exc}"}
        return episode


def collect_rollout(env_factory: Callable[[], CardEnv], seeds: list[int],
                    policy: Policy, *, max_steps: int = 10000
                    ) -> list[Episode]:
    """Collect independent episodes; invalid episodes remain diagnostics only."""
    return [collect_episode(env_factory(), seed, policy, max_steps=max_steps)
            for seed in seeds]
