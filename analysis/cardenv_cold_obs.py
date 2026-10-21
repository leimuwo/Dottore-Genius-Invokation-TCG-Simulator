"""Diagnostic: force a new perspective encoding AND rebuild candidates per sample."""
import json
from pathlib import Path
import statistics
import sys
import time
import random

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'analysis'))
from cardenv_acceptance import decks, game_config, CardEnv, provenance, write_json


def main():
    pool = decks()
    snapshots = []
    for game_id in range(0, 180, 10):
        a, b, seed = game_config(game_id, pool)
        env = CardEnv()
        obs = env.reset(a[1], b[1], seed)
        policy = random.Random(200000 + game_id // 2)
        while not env.done:
            if obs.stage_index == 0 and obs.decision_index % 7 == 0:
                snapshots.append(env.fork())
            out = env.step(policy.randrange(len(obs.choices)))
            obs = out.obs
    rounds = []
    for _ in range(3):
        timings = []
        for snapshot in snapshots:
            env = snapshot.fork()
            # Diagnostic only: explicitly bypass both caches without altering the state.
            env._enc_cache = {}
            env._choices_cache = None
            t = time.perf_counter()
            env.obs()
            timings.append((time.perf_counter() - t) * 1000)
        rounds.append(dict(samples=len(timings), mean_ms=statistics.mean(timings),
                           p95_ms=sorted(timings)[int(len(timings) * .95)], max_ms=max(timings)))
    report = dict(rounds=rounds, passed=all(r['mean_ms'] <= 1 for r in rounds),
                  source_sha256=provenance()['source_sha256'],
                  method='18 full games across 9 decks; every seventh fresh decision; '
                         'both observation and choice caches cleared before timing full obs()')
    write_json(Path(sys.argv[1]), report)
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
