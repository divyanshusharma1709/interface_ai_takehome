"""Shared miner pieces: the candidate record, source helpers, and the Miner base."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..core.textutil import MD_IMAGE, SENT_BREAK_CAPITAL, strip_md_links
from ..core.types import SourceRef
from ..ingest.facts import AUTHORITY
from ..site_profile import ORIGIN

MAX_SOURCES = 5                    # a conflict cites at most this many pages
QUOTE_MAX = 200                    # characters of page text kept as a source quote
EDITORIAL = ("newsroom", "blog")   # dated stories, not the site's standing content
STOP_WORDS = frozenset("""a an and are as at be been by can do for from has have had how i if in is it its of on or
our so than that the their them then there these they this to us was we were what when where which who will
with would you your not no also""".split())
BULLET = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


@dataclass
class RateInfo:
    """What folding and ordering need from a rate conflict."""
    fold_key: tuple                # (snapshot tag, product + qualifiers): a row's rate / APR / APY conflicts share it
    attribute: str
    purpose: str | None
    loser_urls: list
    core: str                      # product name and the rest of the topic after the attribute
    tail: str
    order: tuple                   # (stale, same-page, business, jumbo, tier): the headline row sorts first


@dataclass
class Cand:
    kind: str                      # 'numeric' | 'stat' | 'link' | 'date'
    topic: str
    values: list
    resolution: str | None
    reason: str
    sources: list
    confidence: float
    salience: float
    key: str                       # unique and stable: the final tie-break of every ordering
    rate: RateInfo | None = field(default=None, kw_only=True)
    # (headline resolution, twin label, twin resolutions) once a twin is absorbed

    @property
    def score(self) -> float:
        return round(self.confidence * self.salience, 4)


def path_of(url: str) -> str:
    return (urlparse(url).path or "/").rstrip("/") or "/"


def strip_origin(url: str) -> str:
    return url.replace(ORIGIN, "")


def clean(text: str) -> str:
    # Page text with images, link targets and emphasis removed.
    return strip_md_links(MD_IMAGE.sub("", text)).replace("**", "").replace("<br>", " ")


def authority_of(pv) -> float:
    return AUTHORITY.get(pv.page_type, 1.0)


def fact_source(f, snapshot_id=None) -> SourceRef:
    return SourceRef(f.raw_url or f.url, snapshot_id or f.snapshot_id, (f.row_text or "")[:QUOTE_MAX])


def append_source(sources: list, ref: SourceRef) -> None:
    # Add a source unless the list is full or already cites that URL.
    if len(sources) < MAX_SOURCES and all(ref.url != s.url for s in sources):
        sources.append(ref)


def pair(kind, topic, winner, loser, reason, sources, conf, sal, key) -> Cand:
    # A candidate with two competing values; the winner is listed first and is the resolution.
    return Cand(kind, topic, [winner, loser], winner, reason, sources, conf, sal, key)


class Miner:
    """One detector of contradictions: subclasses implement run(); pages() and sentences() are the steps they share."""

    def __init__(self, kb):
        self.kb = kb

    def run(self) -> list:
        raise NotImplementedError

    @staticmethod
    def pages(view, skip_editorial=True, html_only=False, skip_calculators=False):
        # (url, page) in URL order; listing pages never, the other kinds as the miner asks.
        for url, pv in sorted(view.pages.items()):
            if pv.parsed.listing or (skip_editorial and pv.page_type in EDITORIAL):
                continue
            if not (html_only and pv.parsed.fmt != "html") and not (skip_calculators and "calculator" in url):
                yield url, pv

    @staticmethod
    def sentences(text, min_len=0, max_len=10 ** 9, skip=(), links=False, bullet=BULLET):
        # Sentences of the prose lines, within the length bounds; `skip` prefixes drop table rows /
        # headings, `links` drops URL lines.
        for line in clean(text).split("\n"):
            line = line.strip()
            if not line or line.startswith(skip) or (links and ("http" in line or "data:" in line)):
                continue
            for sent in SENT_BREAK_CAPITAL.split(bullet.sub("", line)):
                if min_len <= len(sent) <= max_len:
                    yield sent
