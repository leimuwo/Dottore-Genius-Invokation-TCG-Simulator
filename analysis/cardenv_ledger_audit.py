"""Independently recompute acceptance counts from the saved event ledgers."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys


def audit(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    seen, completed = set(), set()
    terminals, seats, pairs = Counter(), Counter(), Counter()
    choices = commits = 0
    for path in sorted(folder.glob('events-*.jsonl.gz')):
        active = None
        expected_step = game_choices = game_commits = 0
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                game_id = event['game_id']
                if event['type'] == 'start':
                    assert active is None and game_id not in seen
                    seen.add(game_id)
                    active = game_id
                    expected_step = game_choices = game_commits = 0
                    seats[(event['deck_p1'], 'P1')] += 1
                    seats[(event['deck_p2'], 'P2')] += 1
                    pairs[(event['deck_p1'], event['deck_p2'])] += 1
                elif event['type'] == 'choice':
                    assert active == game_id and event['step'] == expected_step
                    assert event['candidates'] == len(event['candidates_content'])
                    candidate = event['candidates_content'][event['index']]
                    assert candidate == [event['label'], event['feature']]
                    assert len({json.dumps(c[1]) for c in event['candidates_content']}) == event['candidates']
                    if not event['done']:
                        assert event['rewards'] == {'P1': 0.0, 'P2': 0.0}
                    else:
                        reason = event['terminal_reason']
                        expected = ({'P1': -1.0, 'P2': -1.0} if reason == 'timeout'
                                    else {'P1': 1.0 if reason == 'kill:P1' else -1.0,
                                          'P2': 1.0 if reason == 'kill:P2' else -1.0})
                        assert event['rewards'] == expected and event['committed']
                    expected_step += 1
                    game_choices += 1
                    game_commits += event['committed']
                elif event['type'] == 'end':
                    assert active == game_id and event['status'] == 'complete'
                    assert event['choices'] == game_choices == event['stats']['attempts']
                    assert game_commits == event['stats']['commits']
                    assert all(event['stats'][k] == 0 for k in ('illegal', 'exception', 'noop'))
                    assert game_id not in completed
                    completed.add(game_id)
                    choices += game_choices
                    commits += game_commits
                    terminals[event['terminal_reason']] += 1
                    active = None
            assert active is None
    assert completed == seen == set(range(manifest['games']))
    assert len(completed) == summary['completed'] == summary['games']
    assert choices == summary['choices']
    assert commits == summary['stats']['commits']
    assert dict(terminals) == summary['terminal_counts']
    names = {name for name, _ in seats}
    assert all(seats[(name, 'P1')] == seats[(name, 'P2')] for name in names)
    assert all(pairs[(a, b)] == pairs[(b, a)] for a, b in pairs)
    for name, expected in summary['ledger_sha256'].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    result = dict(passed=True, games=len(completed), choices=choices, commits=commits,
                  terminal_counts=dict(terminals), decks=len(names), ordered_matchups=len(pairs),
                  seat_counts={name: {side: seats[(name, side)] for side in ('P1', 'P2')} for name in sorted(names)},
                  all_candidates_match_recorded_index=True, candidate_features_unique=True,
                  both_seats_balanced=True, ledger_checksums_match=True)
    (folder / 'ledger-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1])), ensure_ascii=False))
