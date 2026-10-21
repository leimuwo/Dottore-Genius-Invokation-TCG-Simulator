"""Command line entry point for the M2 baseline smoke run."""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .loop import train_steps
from .policy import PolicyValueNet
from .ppo import PPOConfig, PPOTrainer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/m2-smoke"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = PolicyValueNet(hidden_dim=args.hidden_dim)
    trainer = PPOTrainer(model, PPOConfig())
    result = train_steps(model, trainer, steps=args.steps, seed=args.seed)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "choice_steps": args.steps,
        "model": {"obs_dim": model.obs_dim, "feature_dim": model.feature_dim,
                  "hidden_dim": model.hidden_dim},
        "reward": {"gamma": 0.99, "shaping_lambda": 1e-2, "shaping": True},
        "result": result.__dict__,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    save_checkpoint(args.output / "checkpoint.pt", trainer, step=result.choice_steps,
                    config=manifest)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
