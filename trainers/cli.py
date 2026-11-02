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

from dgisim_ext.env_api import CardEnv
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
    parser.add_argument("--shaping-lambda", type=float, default=1e-2)
    parser.add_argument("--no-shaping", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--env-workers", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.env_workers <= 0:
        parser.error("--env-workers must be positive")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be non-negative")
    torch.set_num_threads(args.torch_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = PolicyValueNet(hidden_dim=args.hidden_dim).to(device)
    trainer = PPOTrainer(model, PPOConfig())
    env_factory = lambda: CardEnv(
        shaping=not args.no_shaping, shaping_lambda=args.shaping_lambda)
    next_checkpoint = args.checkpoint_interval
    def checkpoint_callback(step, current_model, current_trainer):
        nonlocal next_checkpoint
        if args.checkpoint_interval and step >= next_checkpoint:
            save_checkpoint(args.output / f"checkpoint-{step}.pt", current_trainer,
                            step=step, config={"checkpoint_interval": args.checkpoint_interval})
            next_checkpoint += args.checkpoint_interval
    result = train_steps(model, trainer, steps=args.steps, seed=args.seed,
                         env_factory=env_factory,
                         rollout_steps=args.rollout_steps,
                         max_episode_steps=args.max_episode_steps,
                         env_workers=args.env_workers,
                         checkpoint_callback=checkpoint_callback)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "env_workers": args.env_workers,
        "seed": args.seed,
        "choice_steps": args.steps,
        "model": {"obs_dim": model.obs_dim, "feature_dim": model.feature_dim,
                  "hidden_dim": model.hidden_dim},
        "reward": {"gamma": 0.99, "shaping_lambda": args.shaping_lambda,
                   "shaping": not args.no_shaping},
        "result": result.__dict__,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    metrics_path = args.output / "train_metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": result.choice_steps,
                                 "metrics": result.metrics}, default=str) + "\n")
    events_path = args.output / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in result.metrics.get("commit_diagnostics", []):
            handle.write(json.dumps({"event": "commit", **event}, default=str) + "\n")
        for reason, count in (("kill", result.metrics.get("kill_rate", 0.0)),
                              ("engine_timeout", result.metrics.get("engine_timeout_rate", 0.0)),
                              ("truncation", result.metrics.get("truncated_rate", 0.0))):
            handle.write(json.dumps({"event": "episode_rate", "reason": reason,
                                     "rate": count}, default=str) + "\n")
    save_checkpoint(args.output / "checkpoint.pt", trainer, step=result.choice_steps,
                    config=manifest)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
