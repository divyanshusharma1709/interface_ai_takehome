"""Chunking of cleaned pages and the pure-Python BM25 index over chunks (page-type weights)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..core.dates import dates_mentioned
from ..core.textutil import SENT_BREAK, strip_md_links
from ..core.tokens import index_view, tokenize
from ..site_profile import strip_title_suffix

# ---- bm25
K1, B = 1.2, 0.75

# Multiplicative score weights by page type: authoritative sources up, noisy/legal sources down.
# Kept deliberately mild and generic (never tuned to individual questions).
TYPE_WEIGHT = {
    "rate_sheet": 1.25,
    "product": 1.15,
    "branch_contact": 1.15,
    "newsroom": 1.0,
    "other": 0.9,
    "blog": 0.8,
    "form_legal": 0.3,
}
LISTING_WEIGHT = 0.7  # tag/author/category index pages
JUNK_TABLE_WEIGHT = 0.5  # PDF-converted pages with |Col2| junk tables


@dataclass
class Doc:
    chunk: Chunk
    weight: float


class BM25Index:
    """BM25 (k1=1.2, b=0.75) over the chunks of one snapshot view."""

    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total = 0
        for i, d in enumerate(docs):
            tf = d.chunk.tf
            total += d.chunk.length
            for term, n in tf.items():
                self.postings[term].append((i, n))
        self.avgdl = total / max(1, len(docs))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> dict[int, float]:
        """Weighted BM25 score for every chunk that matches at least one query term."""
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                dl = self.docs[i].chunk.length
                out[i] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / self.avgdl))
        return {i: s * self.docs[i].weight for i, s in out.items()}


# ---- chunks
MAX_CHARS = 1400  # soft cap for prose chunks
TABLE_TAIL_MAX = 2600
TABLE_MAX = 3500
MIN_CHARS = 250

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_CHUNK_URL = re.compile(r"https?://[^\s)\]>\"']+")


@dataclass(eq=False)
class Chunk:
    """A section-level passage."""

    page_url: str
    ordinal: int
    heading_path: str
    text: str
    page_type: str
    dates: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    source_url: str = ""
    _tf: Counter | None = field(default=None, repr=False)
    _len: int = field(default=0, repr=False)

    @property
    def length(self) -> int:
        self.tf
        return self._len

    @property
    def index_text(self) -> str:
        """Contextual prefix (heading path) + text, the string that gets indexed."""
        return f"{self.heading_path}\n{index_view(self.text)}"

    @property
    def tf(self) -> Counter:
        if self._tf is None:
            toks = tokenize(self.index_text)
            self._tf = Counter(toks)
            self._len = len(toks)
        return self._tf


def _plain(s: str) -> str:
    return " ".join(strip_md_links(s).split())


def _is_table(block: list[str]) -> bool:
    return block[0].startswith("|")


def _is_caption(block: list[str]) -> bool:
    return (
        len(block) == 1
        and not block[0].startswith(("|", "-", "#", "[", ">"))
        and len(block[0]) <= 80
        and not block[0].endswith(".")
    )


def _blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    cur: list[str] = []
    for l in lines:
        if not l.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if cur and (l.startswith("|")) != (cur[0].startswith("|")):
            blocks.append(cur)
            cur = []
        cur.append(l)
    if cur:
        blocks.append(cur)
    return blocks


def _split_table(rows: list[str]) -> list[list[str]]:
    if sum(len(r) + 1 for r in rows) <= TABLE_MAX or len(rows) < 4:
        return [rows]
    header, body = rows[:2], rows[2:]
    parts, cur, size = [], [], sum(len(r) for r in header)
    for r in body:
        if cur and size + len(r) > TABLE_MAX:
            parts.append(header + cur)
            cur, size = [], sum(len(h) for h in header)
        cur.append(r)
        size += len(r) + 1
    if cur:
        parts.append(header + cur)
    return parts


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for s in SENT_BREAK.split(text):
        if cur and len(cur) + len(s) + 1 > MAX_CHARS:
            parts.append(cur)
            cur = ""
        while len(s) > MAX_CHARS:  # unbroken run: hard split
            parts.append(s[:MAX_CHARS])
            s = s[MAX_CHARS:]
        cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def _pack_section(path: list[str], blocks: list[list[str]]) -> list[tuple[list[str], str, bool]]:
    out: list[tuple[list[str], str, bool]] = []
    buf: list[str] = []
    absorbing = False

    def flush() -> None:
        nonlocal buf
        if buf:
            out.append((path, "\n\n".join(buf), False))
            buf = []

    def caption_ahead(i: int) -> bool:
        if not _is_caption(blocks[i]):
            return False
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        if nxt and _is_table(nxt):
            return True
        nn = blocks[i + 2] if i + 2 < len(blocks) else None
        return bool(nxt and nn and _is_caption(nxt) and _is_table(nn))

    for i, b in enumerate(blocks):
        if _is_table(b):
            caps: list[str] = []
            # captions: up to 2 short label lines immediately before the table
            while len(caps) < 2 and buf and "\n" not in buf[-1] and _is_caption([buf[-1]]):
                caps.insert(0, buf.pop())
            flush()
            for part in _split_table(b):
                text = "\n".join(caps + [""] + part) if caps else "\n".join(part)
                out.append((path + [_plain(c) for c in caps], text.replace("\n\n|", "\n|"), True))
            absorbing = True
            continue
        text = "\n".join(b)
        if absorbing:
            if not caption_ahead(i) and len(out[-1][1]) + len(text) + 2 <= TABLE_TAIL_MAX:
                out[-1] = (out[-1][0], out[-1][1] + "\n\n" + text, True)
                continue
            absorbing = False
        for piece in _split_long(text):
            if buf and sum(len(x) + 2 for x in buf) + len(piece) > MAX_CHARS:
                flush()
            buf.append(piece)
    flush()
    return out


def chunk_page(url: str, title: str, page_type: str, text: str, source_url: str = "") -> list[Chunk]:
    """Split a cleaned page into heading-scoped chunks with contextual heading paths."""
    root = strip_title_suffix(title).strip() or url
    stack: list[tuple[int, str]] = []
    sections: list[tuple[list[str], list[str]]] = []
    cur: list[str] = []

    def path() -> list[str]:
        p = [root] + [h for _, h in stack]
        return [x for i, x in enumerate(p) if i == 0 or x != p[i - 1]]  # drop h1 == title duplicates

    cur_path = path()
    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            sections.append((cur_path, cur))
            level, head = len(m.group(1)), _plain(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, head))
            cur_path, cur = path(), []
        else:
            cur.append(line)
    sections.append((cur_path, cur))

    packed: list[tuple[list[str], str, bool]] = []
    for p, lines in sections:
        packed.extend(_pack_section(p, _blocks(lines)))

    # merge tiny prose chunks into the following chunk (keeping the tiny chunk's heading as a line)
    merged: list[tuple[list[str], str, bool]] = []
    carry = ""
    for p, t, is_tbl in packed:
        if carry:
            t, carry = carry + "\n\n" + t, ""
        if not is_tbl and len(t) < MIN_CHARS:
            carry = (p[-1] + "\n" if len(p) > 1 else "") + t
            continue
        merged.append((p, t, is_tbl))
    if carry:
        if merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n" + carry, merged[-1][2])
        else:
            merged.append((cur_path, carry, False))

    chunks = []
    for i, (p, t, _) in enumerate(merged):
        urls = tuple(dict.fromkeys(u.rstrip(".,;") for u in _CHUNK_URL.findall(t)))[:20]
        chunks.append(Chunk(url, i, " > ".join(p), t, page_type, dates_mentioned(t), urls, source_url))
    return chunks
