"""`conflicts()`: places where the website contradicts itself, which value is right, and why."""
# Winner: numeric facts use `rank_key` so `answer()` and `conflicts()` agree; prose uses the newest dated page;
# links the most authoritative linking page. Candidates are sorted by confidence x salience, cut at the cap,
# then put in canonical order.
from __future__ import annotations

from collections import defaultdict

from . import vocab as V
from .base import append_source
from .dates import PeriodMiner
from .numeric import NO_TIER, NumericMiner, ParallelMiner
from .stats import StatMiner

DEFAULT_CAP = 85            # conflicts returned by default: every real contradiction without burying them in noise


def fold_rate_twins(cands):
    # A rate conflict and the APR (or APY and dividend-rate) conflict of the same row, snapshot and pages are
    # one contradiction.
    groups = defaultdict(list)
    for c in sorted(cands, key=lambda c: c.key):
        if c.rate is not None:
            groups[c.rate.fold_key].append(c)
    absorbed = set()
    for _, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for head_attr, twin_attr, tag, label in V.RATE_PAIRS:
            heads = [c for c in group if c.rate.attribute == head_attr]
            same_pages = lambda a, b: bool(set(a.rate.loser_urls) & set(b.rate.loser_urls))
            taken = defaultdict(list)
            for twin in (c for c in group if c.rate.attribute == twin_attr):
                head = next((h for h in heads if h.rate.purpose == twin.rate.purpose and same_pages(h, twin)), None) \
                    or next((h for h in heads if same_pages(h, twin)), None)
                if head is not None:
                    taken[id(head)].append(twin)
                    absorbed.add(id(twin))
            for head in heads:
                twins = taken[id(head)]
                if twins:
                    head.topic = f"{head.rate.core} {tag}{head.rate.tail}"
                    shown = f"{head.resolution} APY" if head_attr == "apy" else head.resolution
                    head.resolution = f"{shown} ({'; '.join(f'{label} {t.resolution}' for t in twins)})"
                    head.reason += (f" The same row's {label} also disagrees: {twins[0].values[0]} vs "
                                    f"{twins[0].values[1] if len(twins[0].values) > 1 else ''}.")
                    for ref in twins[0].sources:
                        append_source(head.sources, ref)
    return [c for c in cands if id(c) not in absorbed]


def canonical_order(c):
    # Latest snapshot first, cross-page before same-page, headline row (personal, standard, lowest tier) first,
    # then confidence.
    return (*(c.rate.order if c.rate else (0, 0, 0, 0, NO_TIER)), -c.confidence, c.key)




class ConflictMiner:
    """Mines the knowledge base once; `conflicts(cap)` returns the top candidates, best first."""

    def __init__(self, kb):
        self.kb = kb
        self._all = None

    def candidates(self):
        if self._all is None:
            cands = fold_rate_twins(self._mine())
            self._all = sorted(cands, key=lambda c: (-c.score, c.key))
        return self._all

    def _mine(self):
        kb = self.kb
        numeric = NumericMiner(kb)
        rate_cands = numeric.run()
        miners = [ParallelMiner(kb), StatMiner(kb), PeriodMiner(kb)]
        return rate_cands + [c for m in miners for c in m.run()]

    def conflicts(self, cap=DEFAULT_CAP):
        # The best `cap` candidates by confidence x salience, in canonical order.
        return sorted(self.candidates()[:cap], key=canonical_order)
