"""Reproducible CardEnv acceptance; JSON reports and compressed event ledgers."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgisim_ext.env_api import CardEnv, CardEnvError, DEFAULT_PLAN
from src.dgisim.deck import FrozenDeck
from src.dgisim.state.enums import Pid


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def decks():
    result = []
    for split in ('train', 'test'):
        for path in sorted((ROOT / 'decks' / split).glob('*.json')):
            deck = FrozenDeck.from_json(path.read_text(encoding='utf-8'))
            if deck is None:
                raise ValueError(f'invalid deck: {path}')
            result.append((f'{split}/{path.stem}', deck))
    if len(result) != 9:
        raise ValueError('expected 6 train +3 test decks')
    return result


def game_config(game_id, pool):
    # Every 162 games visits all 81 ordered matchups and their seat reversals.
    pair = (game_id // 2) % (len(pool) ** 2)
    a, b = divmod(pair, len(pool))
    if game_id % 2:
        a, b = b, a
    return pool[a], pool[b], 100_000 + game_id // 2


def rewards_json(rewards):
    return {p.name: rewards[p] for p in Pid}


def episode(game_id, pool, events=None, check_observations=False, collect_perf=False):
    a, b, seed = game_config(game_id, pool)
    policy_seed = 200_000 + game_id // 2
    policy = random.Random(policy_seed)
    env = CardEnv()
    row = dict(game_id=game_id, engine_seed=seed, policy_seed=policy_seed,
               deck_p1=a[0], deck_p2=b[0])
    h = hashlib.sha256()
    steps = 0
    cpu_start, start = time.process_time(), time.perf_counter()
    cold, hot = [], []
    try:
        obs = env.reset(a[1], b[1], seed)
        if events:
            events.write(dumps(dict(type='start', **row)) + '\n')
        while not env.done:
            if steps > 30_000:
                raise RuntimeError('episode choice limit')
            if collect_perf:
                t = time.perf_counter()
                env.obs()
                hot.append((time.perf_counter() - t) * 1000)
            idx = policy.randrange(len(obs.choices))
            choice = obs.choices[idx]
            event = dict(type='choice', game_id=game_id, step=steps,
                         actor=obs.seat.name, decision=obs.decision_index,
                         stage=obs.stage_index, index=idx, label=choice.label,
                         feature=choice.feature, candidates=len(obs.choices))
            event['candidates_content'] = [(c.label, c.feature) for c in obs.choices]
            event['prefix'] = obs.meta
            if check_observations:
                event['views'] = {p.name: digest(env.obs(p).enc) for p in Pid}
            old_encodes, old_time = env.stats['encodes'], env.stats['encode_seconds']
            out = env.step(idx)
            if collect_perf and env.stats['encodes'] > old_encodes:
                cold.append((env.stats['encode_seconds'] - old_time) * 1000)
            event.update(rewards=rewards_json(out.info['rewards']), done=out.done,
                         committed=out.info['committed'])
            if out.done:
                event['terminal_reason'] = out.terminal_reason
            serialized = dumps(event)
            if check_observations:
                h.update(serialized.encode())
            if events:
                events.write(serialized + '\n')
            obs = out.obs
            steps += 1
        row.update(status='complete', terminal_reason=out.terminal_reason,
                   rewards=rewards_json(out.info['rewards']))
    except Exception as exc:
        row.update(status='failed', category=getattr(exc, 'category', 'exception'),
                   error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    row.update(choices=steps, stats=env.stats, wall_s=time.perf_counter() - start,
               cpu_s=time.process_time() - cpu_start)
    if check_observations:
        row['trajectory_sha256'] = h.hexdigest()
    if collect_perf:
        row['cold_ms'] = cold
        row['hot_ms'] = hot
    if events:
        events.write(dumps(dict(type='end', **row)) + '\n')
    return row


def shard(args):
    shard_id, workers, count, directory = args
    pool = decks()
    folder = Path(directory)
    rows = []
    with gzip.open(folder / f'events-{shard_id:02}.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1) as events:
        with (folder / f'games-{shard_id:02}.jsonl').open('w', encoding='utf-8') as games:
            for game_id in range(shard_id, count, workers):
                row = episode(game_id, pool, events)
                games.write(dumps(row) + '\n')
                games.flush()
                rows.append(row)
                if len(rows) % 25 == 0:
                    write_json(folder / f'progress-{shard_id:02}.json',
                               dict(completed=len(rows), failures=sum(r['status'] != 'complete' for r in rows)))
    return rows


def provenance():
    paths = sorted(set([*ROOT.glob('src/dgisim/**/*.py'), *ROOT.glob('dgisim_ext/*.py'),
                        *ROOT.glob('decks/*/*.json'), Path(__file__)]))
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                files=files, source_sha256=digest(files), python=sys.version,
                platform=platform.platform(), cpu_count=os.cpu_count())


def summarize(rows, wall):
    complete = [r for r in rows if r['status'] == 'complete']
    counts = Counter(r.get('terminal_reason', 'failed') for r in rows)
    stats = {k: sum(r['stats'][k] for r in rows) for k in rows[0]['stats']}
    attempts = max(1, stats['attempts'])
    return dict(games=len(rows), completed=len(complete), failed=len(rows) - len(complete),
                terminal_counts=dict(counts), timeout_rate=counts['timeout'] / max(1, len(complete)),
                choices=sum(r['choices'] for r in rows), stats=stats,
                illegal_rate=stats['illegal'] / attempts, exception_rate=stats['exception'] / attempts,
                noop_rate=stats['noop'] / attempts,
                wall_s=wall, cpu_s=sum(r['cpu_s'] for r in rows),
                games_per_s=len(complete) / wall, choices_per_s=sum(r['choices'] for r in rows) / wall,
                commits_per_s=stats['commits'] / wall,
                encode_ms=1000 * stats['encode_seconds'] / max(1, stats['encodes']),
                failures=[r for r in rows if r['status'] != 'complete'])


def determinism_child(ids):
    pool = decks()
    rows = [episode(i, pool, check_observations=True) for i in ids]
    return [{k: r.get(k) for k in ('game_id', 'status', 'trajectory_sha256', 'error')} for r in rows]


def deterministic_check(count):
    # Across all 9 decks; choose different opponents too.
    ids = list(range(0, min(count, 18) * 10, 10))
    local = determinism_child(ids)
    child_env = dict(os.environ, PYTHONHASHSEED='987654')
    remote = json.loads(subprocess.check_output(
        [sys.executable, '-B', str(Path(__file__).resolve()), '--child', ','.join(map(str, ids))],
        cwd=ROOT, text=True, encoding='utf-8', env=child_env))
    return dict(passed=local == remote and all(r['status'] == 'complete' for r in local),
                games=len(local), parent=local, child=remote)


def percentile(values, q):
    return sorted(values)[min(len(values) - 1, int(len(values) * q))] if values else None


def benchmark():
    pool = decks()
    # Warm all paths before timing. No other benchmark runs concurrently.
    episode(0, pool)
    game_ids = [i * 10 for i in range(18)]
    rounds = []
    for repeat in range(3):
        start = time.perf_counter()
        rows = [episode(i, pool, collect_perf=True) for i in game_ids]
        report = summarize(rows, time.perf_counter() - start)
        cold = [v for r in rows for v in r['cold_ms']]
        hot = [v for r in rows for v in r['hot_ms']]
        report.update(cold_mean_ms=statistics.mean(cold), cold_p95_ms=percentile(cold, .95),
                      hot_mean_ms=statistics.mean(hot), hot_p95_ms=percentile(hot, .95), repeat=repeat)
        report['estimated_100k_seconds'] = 100_000 / report['choices_per_s']
        report['passed'] = (not report['failed'] and report['cold_mean_ms'] <= 1
                            and report['estimated_100k_seconds'] <= 300)
        rounds.append(report)
    actual = timed_rollout(100_000, pool)
    return dict(passed=all(r['passed'] for r in rounds) and actual['passed'], rounds=rounds,
                actual_100k=actual,
                note='round estimates are extrapolations; actual_100k times 100,000 real choice steps')


def timed_rollout(count, pool):
    policy = random.Random(300_000)
    env = CardEnv()
    game_id = 0
    a, b, seed = game_config(game_id, pool)
    obs = env.reset(a[1], b[1], seed)
    start = time.perf_counter()
    for _ in range(count):
        out = env.step(policy.randrange(len(obs.choices)))
        if out.done:
            game_id += 1
            a, b, seed = game_config(game_id, pool)
            obs = env.reset(a[1], b[1], seed)
        else:
            obs = out.obs
    wall = time.perf_counter() - start
    return dict(choices=count, wall_s=wall, choices_per_s=count / wall,
                complete_games=game_id, passed=wall <= 300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--determinism', action='store_true')
    parser.add_argument('--child')
    args = parser.parse_args()
    if args.child:
        print(dumps(determinism_child([int(i) for i in args.child.split(',')])))
        return 0
    if args.games <= 0 or args.workers <= 0:
        parser.error('games and workers must be positive')
    folder = args.out or ROOT.parent / 'analysis' / ('cardenv-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), games=args.games,
                    workers=args.workers, sampling='81 ordered pairs, paired seat reversal; validation only',
                    policy='uniform per choice layer, not uniform over complete actions',
                    timeout_penalty=-1, shaping=False, provenance=provenance())
    write_json(folder / 'manifest.json', manifest)
    if args.determinism:
        result = deterministic_check(18)
        write_json(folder / 'determinism.json', result)
        print(dumps(dict(determinism_passed=result['passed'])), flush=True)
        if not result['passed']:
            return 1
    if args.benchmark:
        result = benchmark()
        write_json(folder / 'benchmark.json', result)
        print(dumps(dict(benchmark=result)), flush=True)
        if not result['passed']:
            return 1
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(shard, (i, args.workers, args.games, str(folder)))
                   for i in range(args.workers)]
        for future in as_completed(futures):
            result = future.result()
            rows.extend(result)
            print(dumps(dict(shard_finished=len(result), collected=len(rows),
                             failures=sum(r['status'] != 'complete' for r in result))), flush=True)
    summary = summarize(sorted(rows, key=lambda r: r['game_id']), time.perf_counter() - start)
    final_provenance = provenance()
    summary['source_unchanged'] = final_provenance['source_sha256'] == manifest['provenance']['source_sha256']
    summary['passed'] = (len(rows) == args.games and summary['failed'] == 0
                         and all(summary[k] == 0 for k in ('illegal_rate', 'exception_rate', 'noop_rate'))
                         and summary['source_unchanged'])
    summary['ledger_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(folder.glob('*.jsonl*'))}
    write_json(folder / 'summary.json', summary)
    print(dumps(summary), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
