"""Reproducible model, optimizer, and RNG checkpoints."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .ppo import PPOTrainer

__all__ = ["save_checkpoint", "load_checkpoint"]


def _rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


def save_checkpoint(path: str | Path, trainer: PPOTrainer, *, step: int,
                    config: dict[str, Any] | None = None) -> None:
    payload = {
        "format": 1,
        "step": int(step),
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "ppo_config": trainer.config.__dict__,
        "config": config or {},
        "rng": _rng_state(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, trainer: PPOTrainer,
                    *, restore_rng: bool = True) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != 1:
        raise ValueError("unsupported checkpoint format")
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return {"step": payload["step"], "config": payload.get("config", {}),
            "ppo_config": payload.get("ppo_config", {})}
