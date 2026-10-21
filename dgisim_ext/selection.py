"""Bounded, autoregressive multiset selection; never enumerate PlayerActions."""
from dataclasses import dataclass, replace
from functools import lru_cache

from src.dgisim.card.cards import Cards
from src.dgisim.dice import ActualDice, AbstractDice
from src.dgisim.element import Element


@lru_cache(maxsize=32768)
def _completion(items, caps, cost, prefix):
    needed = cost.num_dice() - sum(prefix)
    remaining = caps[len(prefix):]
    if needed < 0 or needed > sum(remaining):
        return False
    if not remaining:
        return ActualDice(dict(zip(items, prefix))).just_satisfy(cost)
    return any(_completion(items, caps, cost, prefix + (n,))
               for n in range(min(remaining[0], needed) + 1))


@dataclass(frozen=True)
class Selection:
    kind: str
    items: tuple
    caps: tuple[int, ...]
    prefix: tuple[int, ...] = ()
    cost: AbstractDice | None = None

    @classmethod
    def dice(cls, raw):
        items = tuple(e for e in Element if raw[e] > 0)
        return cls('dice', items, tuple(raw[e] for e in items))

    @classmethod
    def payment(cls, raw, cost):
        return replace(cls.dice(raw), kind='cost', cost=cost)

    @classmethod
    def cards(cls, raw):
        items = tuple(sorted((c for c in raw if raw[c] > 0), key=lambda c: c.__name__))
        return cls('card', items, tuple(raw[c] for c in items))

    @property
    def done(self):
        return len(self.prefix) == len(self.items)

    def counts(self):
        if self.done:
            return ()
        counts = range(self.caps[len(self.prefix)] + 1)
        if self.cost is None:
            return tuple(counts)
        return tuple(n for n in counts
                     if _completion(self.items, self.caps, self.cost, self.prefix + (n,)))

    def choose(self, count):
        if count not in self.counts():
            raise ValueError('invalid multiset count')
        return replace(self, prefix=self.prefix + (count,))

    def result(self):
        if not self.done:
            raise ValueError('selection is incomplete')
        values = {item: n for item, n in zip(self.items, self.prefix) if n}
        return Cards(values) if self.kind == 'card' else ActualDice(values)
