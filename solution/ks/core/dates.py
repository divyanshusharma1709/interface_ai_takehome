"""Months and dates: names, numbers, formatting, and extraction of dates from page text and question text."""

from __future__ import annotations

import re
from datetime import date

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTH_NUMBER = {name[:3].lower(): i for i, name in enumerate(MONTH_NAMES, 1)}  # 'jan' -> 1 ... 'dec' -> 12

# A month as a capturing group: full names or the usual abbreviations ('Sept'); matches case-sensitively as written
# (Title Case) or with re.I.
MONTH_PATTERN = (
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
# Looser: a month abbreviation followed by any letters ('Septmber' still matches), no group.
MONTH_WORD = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"

LONG = re.compile(rf"\b{MONTH_PATTERN}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")  # September 1, 2026
MONTH_YEAR = re.compile(rf"\b{MONTH_PATTERN}\.?,?\s+(\d{{4}})\b")  # September 2026
NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")  # 09/01/2026
ISO = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
_EFF = re.compile(r"\beffective\b", re.I)
_PATH_DATE = re.compile(r"/(20\d\d)/(\d\d)/(\d\d)/")


def month_number(word: str) -> int:
    """'Sept.' / 'september' -> 9 (by the first three letters)."""
    return MONTH_NUMBER[word.lower()[:3]]


def month_day_year(iso: str) -> str:
    """'2026-06-15' -> 'June 15, 2026'."""
    try:
        d = date.fromisoformat(iso[:10])
    except ValueError:
        return iso
    return f"{MONTH_NAMES[d.month - 1]} {d.day}, {d.year}"


def parse_written_date(text: str) -> str | None:
    """ISO date of a date written alone ('May 31, 2026' or '5/31/26'); None for anything else."""
    t = text.strip().rstrip(".")
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", t)
    if m and m.group(1)[:3].lower() in MONTH_NUMBER:
        return f"{int(m.group(3)):04d}-{month_number(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})$", t)
    if m:
        y = int(m.group(3))
        return f"{y + 2000 if y < 100 else y:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def iso_date(y: int, m: int, d: int) -> str | None:
    if y < 100:
        y += 2000
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def first_date(s: str) -> str | None:
    """Earliest-positioned full date (numeric, long-form or ISO) in `s`."""
    best = None  # (position, iso)
    for rx, conv in (
        (NUM, lambda m: iso_date(int(m[3]), int(m[1]), int(m[2]))),
        (LONG, lambda m: iso_date(int(m[3]), month_number(m[1]), int(m[2]))),
        (ISO, lambda m: iso_date(int(m[1]), int(m[2]), int(m[3]))),
    ):
        m = rx.search(s)
        if m and conv(m) and (best is None or m.start() < best[0]):
            best = (m.start(), conv(m))
    return best[1] if best else None


def dates_mentioned(text: str, cap: int = 12) -> tuple[str, ...]:
    """All dates in `text` as ISO strings (YYYY-MM-DD, or YYYY-MM for month+year), in order, unique."""
    found: list[tuple[int, str]] = []
    for m in NUM.finditer(text):
        iso = iso_date(int(m[3]), int(m[1]), int(m[2]))
        if iso:
            found.append((m.start(), iso))
    for m in LONG.finditer(text):
        iso = iso_date(int(m[3]), month_number(m[1]), int(m[2]))
        if iso:
            found.append((m.start(), iso))
    for m in ISO.finditer(text):
        iso = iso_date(int(m[1]), int(m[2]), int(m[3]))
        if iso:
            found.append((m.start(), iso))
    long_spans = [m.span() for m in LONG.finditer(text)]
    for m in MONTH_YEAR.finditer(text):
        if not any(a <= m.start() < b for a, b in long_spans):
            found.append((m.start(), f"{int(m[2]):04d}-{month_number(m[1]):02d}"))
    out: list[str] = []
    for _, iso in sorted(found):
        if iso not in out:
            out.append(iso)
    return tuple(out[:cap])


def effective_dates(text: str) -> list[str]:
    """Dates stated right after the word 'effective' (e.g."""
    out = []
    for m in _EFF.finditer(text):
        d = first_date(text[m.end() : m.end() + 40])
        if d and d not in out:
            out.append(d)
    return sorted(out)


def pick_effective(dates: list[str], fetched_on: str) -> str | None:
    """Page-level effective date: newest stated date that is not after the crawl; else None."""
    ok = [d for d in dates if d <= fetched_on]
    return max(ok) if ok else None


def publish_date(text: str, url_path: str) -> str | None:
    """Best-effort publish date: first long-form date in the first ~700 chars (byline), else /YYYY/MM/DD/ in URL."""
    d = first_date(text[:700])
    if d:
        return d
    m = _PATH_DATE.search(url_path)
    return iso_date(int(m[1]), int(m[2]), int(m[3])) if m else None


def parse_as_of(value) -> str | None:
    """ISO date string from an `as_of` value (ISO date/timestamp, 'May 19, 2026', '05/19/2026')."""
    if not value:
        return None
    v = str(value).strip()
    try:
        return date.fromisoformat(v[:10]).isoformat()
    except ValueError:
        pass
    return first_date(v)


def days_between(a: str, b: str) -> int:
    """Whole days from ISO date `a` to `b` (0 when either is not a date)."""
    try:
        return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days
    except ValueError:
        return 0
