"""Date contradictions: program periods and deadlines stated differently."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from ..core.dates import MONTH_NUMBER
from ..core.textutil import SENT_BREAK_CAPITAL
from ..core.types import SourceRef
from . import vocab as V
from .base import Miner, authority_of, clean, pair, path_of

LONG_LINE = 600                    # a longer line is a paragraph, read a sentence at a time


# ---- program periods and deadlines
def label_tokens(text):
    return frozenset(w.rstrip("s") for w in re.findall(r"[a-z]+",
        text.lower()) if w not in V.PERIOD_GENERIC_WORDS and len(w) >= 3)


@dataclass
class _Period:
    url: str
    pv: object
    sid: str
    label_h: frozenset     # words of the heading above the statement
    label_s: frozenset     # words of the URL, used when there is no useful heading
    name: str
    kind: str              # 'start' | 'end'
    date: tuple            # (month, day, year or 0)
    raw: str               # as displayed, with the year filled in when the line gives it once
    taken_on: str

    def iso(self):
        return f"{self.date[2]:04d}-{self.date[0]:02d}-{self.date[1]:02d}"


def same_program(a, b):
    return any(x and y and (x <= y or y <= x or len(x & y) / min(len(x), len(y)) >= 0.5)
               for x in (a.label_h, a.label_s) for y in (b.label_h, b.label_s))


def dates_in(line):
    # [((month, day, year or 0), as displayed)]; a year written once applies to the dates before it.
    found, year = [], 0
    for m in reversed(list(V.PERIOD_DATE.finditer(line))):
        year = int(m.group(3)) if m.group(3) else year
        found.append((m, year))
    return [((MONTH_NUMBER.get(m.group(1)[:3].lower(), 0), int(m.group(2)), y),
        f"{m.group(1)} {m.group(2)}" + (f", {y}" if y else ""))
            for m, y in reversed(found)]


def statement_kinds(line, dates):
    low = line.lower()
    if (len(dates) >= 2 and ("period" in low or "–" in line or "-" in line)
        and "deadline" not in low.split(":")[0][-12:]):
        return [("start", dates[0]), ("end", dates[-1])]
    if re.search(V.PERIOD_START_WORDS, low) and "deadline" not in low:
        return [("start", dates[0])]
    return [("end", dates[-1])]


class PeriodMiner(Miner):
    """A program's period / deadline stated differently at once, or a deadline before the announced period."""

    def run(self):
        by_snap = defaultdict(list)
        for v in self.kb.views:
            for url, pv in self.pages(v):
                for p in self._page_periods(url, pv, v):
                    by_snap[p.sid].append(p)
        found = {}
        for sid, items in sorted(by_snap.items()):
            heads = defaultdict(set)
            for p in items:
                if p.label_h:
                    heads[p.url].add(p.label_h)
            for i, a in enumerate(items):
                for b in items[i + 1:]:
                    ya, yb = a.date[2], b.date[2]
                    if a.url != b.url and a.kind == b.kind and same_program(a,
                        b) and not (a.date[:2] == b.date[:2] and (not ya or not yb or ya == yb)):
                        found[(a.kind, tuple(sorted((a.raw, b.raw))))] = (sid, a, b)
            for a in (p for p in items if p.kind == "end" and p.date[2]):
                for b in items:
                    if (b.url == a.url and len(heads[a.url]) <= 1 and b.kind == "start" and b.date[2]
                        and b.iso() > a.iso()):
                        found[("order", a.url, a.raw, b.raw)] = (sid, a, b)
        return [self._order_cand(key, a, b) if key[0] == "order" else self._disagree_cand(key, sid, a, b)
                for key, (sid, a, b) in sorted(found.items(), key=lambda kv: str(kv[0]))]

    @staticmethod
    def _page_periods(url, pv, v):
        slug = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")
        lines = []
        for raw_line in pv.parsed.text.split("\n"):
            lines += SENT_BREAK_CAPITAL.split(raw_line) if len(raw_line) > LONG_LINE else [raw_line]
        head, res = "", []
        for line in lines:
            if line.startswith("#"):
                head = line.strip("# ").strip()
            ds = dates_in(clean(line)) if V.PERIOD_CUE.search(line) else []
            if not ds or len(line) > LONG_LINE:
                continue
            hh = head if head and not V.PERIOD_CUE.search(head) and not V.PERIOD_DATE.search(head) else ""
            for kind, (dt, raw) in statement_kinds(line, ds):
                res.append(_Period(url, pv, v.snapshot_id, label_tokens(hh),
                    frozenset() if hh else label_tokens(slug), hh or slug.title(),
                                   kind, dt, raw, v.taken_on))
        return res

    @staticmethod
    def _order_cand(key, a, b):
        title = (b.pv.title or "").split("|")[0].strip()
        reason = (f"{path_of(a.url)} still states the deadline {a.raw} although the same page announces an "
                  f"application period "
                  f"starting {b.raw}; the deadline lies before the period it belongs to, so the newer period wins.")
        return pair("date", f"{label_tokens(b.pv.title or '') and title or b.name} application deadline vs "
                            f"application period", b.raw, a.raw,
                    reason, [SourceRef(b.pv.raw_url, b.sid, b.raw), SourceRef(a.pv.raw_url, a.sid,
                        a.raw)], 0.75, 0.85, f"per:{key}")

    @staticmethod
    def _disagree_cand(key, sid, a, b):
        what = "period start" if a.kind == "start" else "deadline"
        name = a.name if len(a.name) >= len(b.name) and a.label_h else (b.name if b.label_h else a.name)
        alive = [x for x in (a, b) if x.date[2] and x.iso() >= a.taken_on]
        if len(alive) == 1:
            win, why = alive[0], "the other date already lies in the past at that crawl, so it is stale"
        else:
            win = max((a, b), key=lambda x: (authority_of(x.pv), x.pv.fetched_at or "", x.url))
            why = "the more authoritative / more recently crawled page"
        lose = b if win is a else a
        kind = "start" if a.kind == "start" else "deadline"
        reason = (f"{path_of(a.url)} and {path_of(b.url)} state different {kind} dates for the same "
                  f"program in snapshot {sid[:2]} ({a.raw} vs {b.raw}); {why}: {win.raw}.")
        return pair("date", f"{name} application {what}", win.raw, lose.raw, reason,
                    [SourceRef(win.pv.raw_url, win.sid, win.raw), SourceRef(lose.pv.raw_url, lose.sid,
                        lose.raw)], 0.7, 0.85, f"per:{key}")
