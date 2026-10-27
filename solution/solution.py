"""Milestone 3 submission: query understanding + fact-store answers, BM25 extractive fallback, no-number refusals."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:                                    # run_eval.py already puts harness/ on sys.path
    from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
except ImportError:                     # standalone use: look for a kit's harness/ in the parents
    _h = next((d / "harness" for d in _HERE.parents if (d / "harness" / "contract.py").exists()), None)
    if _h is not None:
        sys.path.insert(0, str(_h))
    from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
from ks.qa.router import Answerer
from ks.conflicts.miner import DEFAULT_CAP, ConflictMiner
from ks.ingest.knowledge_base import KnowledgeBase


class Milestone3:
    def __init__(self):
        self.kb = KnowledgeBase()
        self._answerer = None
        self._miner = None

    def ingest(self, snapshot: Snapshot) -> None:
        self.kb.ingest(snapshot)
        self._answerer = None                           # rebuilt lazily so caches never outlive a new snapshot
        self._miner = None

    def answer(self, question: str, as_of=None) -> Answer:
        if self._answerer is None:
            self._answerer = Answerer(self.kb)
        r = self._answerer.answer(question, as_of)
        return Answer(
            text=r.text,
            sources=[Source(url=s.url, snapshot_id=s.snapshot_id, quote=s.quote) for s in r.sources],
            confident=r.confident,
            conflict_note=r.conflict_note,
        )

    def conflicts(self, cap: int = DEFAULT_CAP) -> list[Conflict]:
        """Ranked real-world contradictions (numeric, repeated statistics, moved links); mined lazily and cached."""
        if self._miner is None:
            self._miner = ConflictMiner(self.kb)
        return [Conflict(topic=c.topic, values=list(c.values), resolution=c.resolution, reason=c.reason,
                         sources=[Source(url=s.url, snapshot_id=s.snapshot_id, quote=s.quote) for s in c.sources])
                for c in self._miner.conflicts(cap)]


def build() -> Milestone3:
    return Milestone3()
