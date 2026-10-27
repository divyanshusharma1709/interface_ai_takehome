"""Chunk text broken down to plain lines, bullets, table rows and sentences, and the page-level signals the sentence scorer uses."""
from __future__ import annotations

import re

from ..core.textutil import MD_IMAGE, MD_LINK_URL, SENT_BREAK_ABBREV, strip_md_links


def plain_text(t: str) -> str:
    """Markdown reduced to words: links to their text, emphasis / heading / quote marks dropped."""

    t = strip_md_links(t)
    return re.sub(r"\s+", " ", re.sub(r"[*_#>`]", "", t)).strip()


def render(text: str) -> str:
    """Markdown -> plain text; link targets stay in parentheses unless they are anchors / tel / mailto."""
    text = MD_IMAGE.sub("", text)

    def link(m: re.Match) -> str:
        t, u = m.group(1).strip(), m.group(2)
        if u.startswith(("tel:", "mailto:", "#")) or not u:
            return t
        if u.rstrip("/").endswith(t.rstrip("/")) or t.startswith("http"):
            return t
        return f"{t} ({u})" if t else u
    text = MD_LINK_URL.sub(link, text)
    text = re.sub(r"[*_`]{1,3}", "", text)
    return " ".join(text.split())


def split_sentences(chunk_text: str) -> list[tuple[str, int]]:
    """`split_with_context` without the heading context."""
    return [(s, g) for s, g, _ in split_with_context(chunk_text)]


def split_with_context(chunk_text: str) -> list[tuple[str, int, str]]:
    """(sentence, group, heading) triples: lines / bullets / table rows / sentences of a chunk as plain text."""
    out: list[tuple[str, int, str]] = []
    label, heading = "", ""
    group, in_list = 0, False
    lines = [ln.strip() for ln in chunk_text.split("\n")]
    for i, line in enumerate(lines):
        if not line:
            after = next((ln for ln in lines[i + 1:] if ln), "")
            # a loose list keeps one group
            in_list = in_list and bool(re.match(r"^[-*•]\s+", after))
            continue
        if set(line) <= set("|-: "):
            continue
        nxt = next((ln for ln in lines[i + 1:] if ln), "")
        if re.fullmatch(r"\*\*[^*]{1,40}\*\*:?", line) or (
                len(line.split()) <= 3 and len(line) <= 24 and not re.search(r"[.!?,;]$", line)
                and re.match(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|\d|\$)", nxt)):
            # 'Lobby' above 'Monday-Friday: 9 a.m.-5 p.m.'
            label = line.strip("*: ")
            continue
        bullet = bool(re.match(r"^[-*•]\s+", line))
        if bullet and not in_list:
            group += 1
        in_list = bullet
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            line = ", ".join(c for c in cells if c)
        line = re.sub(r"^[-*•]\s+", "", line)
        if re.match(r"^#{1,6}\s+", line):
            heading = render(re.sub(r"^#{1,6}\s+", "", line)).strip()
        line = re.sub(r"^#{1,6}\s+", "", line)
        if label:
            line, label = f"{label}: {line}", ""
        plain = render(line)
        for s in SENT_BREAK_ABBREV.split(plain):
            s = s.strip()
            if len(s) >= 8:
                out.append((s, group if bullet else -1, heading if bullet else ""))
    return out


_TEASER_HEAD = re.compile(r"#{2,6} \[[^\]\n]+\]\((https?://[^)\s]+)\)\s*$")


def _same_page(a: str, b: str) -> bool:
    norm = lambda u: u.lower().split("#")[0].rstrip("/").replace("://www.", "://")
    return norm(a) == norm(b)


def is_teaser(page_text: str, sent: str, page_url: str) -> bool:
    at = page_text.find(sent[:50])
    if at <= 0:
        return False
    m = _TEASER_HEAD.search(page_text[max(0, at - 400):at])
    return bool(m) and not _same_page(m.group(1), page_url)

# ---- Page-level signals the sentence scorer uses: which year a release is about, and how much a page type is trusted.
# product and rate pages over articles, listings and legal forms
PAGE_TYPE_TRUST = {
    "rate_sheet": 1.15,
    "product": 1.2,
    "branch_contact": 1.15,
    "newsroom": 0.95,
    "other": 0.9,
    "blog": 0.8,
    "form_legal": 0.4,
}
_YEAR_PHRASE = re.compile(r"\b(?:for|in|during|of|throughout|saved)\s+(20\d\d)\b")


class PageYears:
    """The year a dated release is about: 'for 2022' / 'in 2025' / 'paid out Oct."""

    def __init__(self):
        self._cache: dict = {}

    def of(self, page) -> int | None:
        key = (page.url, page.snapshot_id)
        if key not in self._cache:
            lead = strip_md_links(page.parsed.text or "")[:900]
            m = _YEAR_PHRASE.search(re.sub(r"^.*?\n\n#", "", lead, count=1, flags=re.S) or lead)
            yr = int(m.group(1)) if m else None
            if yr is None and (page.publish_date or "")[:4].isdigit():
                yr = int(page.publish_date[:4])
            self._cache[key] = yr if any(d in page.url for d in ("/newsroom/", "/learn/")) else None
        return self._cache[key]


def page_authority(page, q) -> float:
    """How much a page is trusted to answer `q`."""
    f = PAGE_TYPE_TRUST.get(page.page_type, 1.0)
    if page.parsed.listing:
        # a listing of articles/products points elsewhere rather than answering
        f *= 0.5
    if q.year_hint:
        if str(q.year_hint) in (page.parsed.text[:3000] if page.parsed.text else "") or (
            page.publish_date or ""
        ).startswith(str(q.year_hint)):
            # the page mentions or was published in the year the question names
            f *= 1.25
    return f

