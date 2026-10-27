"""Dated official statistics: "As of: May 31, 2026 | Members: 182,627 | Deposits: $3.0 billion | Assets: $3.6 billion"
boxes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.dates import parse_written_date

_ASOF = re.compile(r"^as of:?\s*(?P<d>[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\s*$", re.I)
_PAIR = re.compile(
    r"^(?P<k>[A-Za-z][A-Za-z ]{2,40}):\s*(?P<v>\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand))?%?)\s*$", re.I
)

# label on the page -> canonical statistic name (and the word used in an answer)
LABELS = {
    "members": "members",
    "membership": "members",
    "deposits": "deposits",
    "loans": "loans",
    "assets": "assets",
    "employees": "employees",
    "branches": "branches",
    "locations": "branches",
    "branch locations": "branches",
}


@dataclass(frozen=True)
class Box:
    date: str  # ISO as-of date
    date_text: str  # as the page writes it
    stats: tuple  # ((canonical name, label, value text), ...)
    url: str
    raw_url: str
    snapshot_id: str
    line: str  # the as-of line, for quoting

    def get(self, name: str):
        return next(((l, v) for n, l, v in self.stats if n == name), None)


def page_boxes(page) -> list[Box]:
    text = page.parsed.text or ""
    if "as of" not in text.lower():
        return []
    lines = [re.sub(r"^[\s|>*#-]+|[\s|*_]+$", "", l) for l in text.split("\n")]
    out = []
    for i, line in enumerate(lines):
        m = _ASOF.match(line)
        iso = parse_written_date(m.group("d")) if m else None
        if not iso:
            continue
        stats, quiet = [], 0
        for nxt in lines[i + 1 : i + 16]:
            pm = _PAIR.match(nxt)
            if pm and pm.group("k").strip().lower() in LABELS:
                stats.append((LABELS[pm.group("k").strip().lower()], pm.group("k").strip(), pm.group("v").strip()))
                quiet = 0
            elif nxt and not pm:
                quiet += 1
                if stats and quiet >= 3:
                    break
        if stats:
            out.append(Box(iso, m.group("d").strip(), tuple(stats), page.url, page.raw_url, page.snapshot_id, line))
    return out


_CACHE: dict[int, list[Box]] = {}


def view_boxes(view) -> list[Box]:
    k = id(view)
    hit = _CACHE.get(k)
    if hit is None or hit[0] is not view:
        boxes = []
        for url, page in sorted(view.pages.items()):
            boxes.extend(page_boxes(page))
        _CACHE.clear()
        _CACHE[k] = (view, boxes)  # type: ignore[assignment]
        hit = _CACHE[k]
    return hit[1]


def best_box(view, name: str):
    """(Box, label, value) with the newest as-of date that states `name` in this snapshot view, or None."""
    best = None
    for b in view_boxes(view):
        got = b.get(name)
        if got and (best is None or (b.date, b.url) > (best[0].date, best[0].url)):
            best = (b, got[0], got[1])
    return best


def box_wins(box: Box, newest_statement: str | None) -> bool:
    """A dated box beats boilerplate whose newest statement is not newer than the box's as-of date."""
    return not newest_statement or box.date >= newest_statement


def forms(value: str) -> list[str]:
    """Normalised surface forms of a statistic value ('182,627', '$3.6 billion', '12.94%')."""
    out = [value]
    m = re.fullmatch(r"(\$?)([\d,]+)(\.\d+)?(\s*(?:billion|million|thousand))?(%?)", value.strip(), re.I)
    if m and m.group(3) and set(m.group(3)[1:]) == {"0"}:
        out.append(f"{m.group(1)}{m.group(2)}{m.group(4) or ''}{m.group(5)}")
    return list(dict.fromkeys(out))
