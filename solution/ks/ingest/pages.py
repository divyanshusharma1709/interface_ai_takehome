"""Page versions: text cleaning (invisible characters, markup, per-snapshot boilerplate) and page parsing."""

from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

from ..core.dates import pick_effective, publish_date
from ..core.textutil import strip_md_links
from ..core.urls import normalize_url
from ..site_profile import (
    BLOG_INDEX_PATHS,
    BLOG_LISTING_PREFIXES,
    BLOG_PREFIXES,
    CONTACT_PATHS,
    LEGAL_PATHS,
    NEWSROOM_INDEX_PATHS,
    NEWSROOM_LISTING_PREFIXES,
    NEWSROOM_PREFIX,
    PRODUCT_PREFIXES,
    RATE_SHEET_PATHS,
)
from .chunks import Chunk

# ---- clean
# Zero-width / bidi / soft-hyphen characters are deleted; NBSP becomes a plain space.
_STRIP = dict.fromkeys([*range(0x200B, 0x2010), *range(0x202A, 0x202F), 0x2060, 0xFEFF, 0x00AD])
_STRIP[0x00A0] = " "

_IMG = re.compile(r'!\[[^\]]*\]\((?:[^()\s]|\([^)]*\))*(?:\s+"[^"]*")?\)')
_LINK = re.compile(r'\[([^\]]*)\]\(([^)\s]*)(?:\s+"[^"]*")?\)')
_EOL_BACKSLASH = re.compile(r"\\{1,2}(?=[ \t]*$)", re.M)
_LITERAL_CRLF = re.compile(r"(?:\\{1,2}[rn])+")  # scraped "\r\n" left as literal text
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!$<>~&])")
_FOOTNOTE = re.compile(r"^\W*\d{1,2}\W*$")
_PCT_FOOTNOTE = re.compile(r"(\d%)\d{1,2}(?=\s|\||$|[,.;:)])")  # "4.20%1" -> "4.20%" (digit glued after %)
_PDF_JUNK = re.compile(r"\|Col\d+\|")

# Phone-country dropdown items from contact forms ("- 🇹🇷 Turkey (Türkiye) +90"): hundreds of lines on ~6% of pages.
_FLAG_ITEM = re.compile(r"^(?:-\s*)?[\U0001F1E6-\U0001F1FF]{2}(?:\s|$)")

# Pure UI chrome that is noise wherever it appears (not frequency based).
_UI_NOISE = re.compile(
    r"^(?:✕|cancel|continue|continue cancel|load more|sort by:.*|#+\s*filter|\* \* \*|---+|"
    r"\d+ min read|loading locations\.*|\d+ results?)$",
    re.I,
)


def strip_invisible(text: str) -> str:
    """Remove zero-width/bidi/soft-hyphen characters and turn NBSP into a space."""
    return text.translate(_STRIP)


def _pdf_table_to_text(text: str) -> str:
    out = []
    for line in text.split("\n"):
        if line.startswith("|"):
            if re.fullmatch(r"\|[\s\-:|]+\|?", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = [c for c in cells if c and not re.fullmatch(r"Col\d+", c)]
            if cells:
                out.append(" | ".join(cells))
        else:
            out.append(line)
    return "\n".join(out)


def precleaned(markdown: str) -> tuple[str, bool]:
    """Page-local cleaning (no cross-page knowledge)."""
    t = strip_invisible(markdown).replace("\r\n", "\n").replace("\r", "\n")
    for _ in range(3):
        u = html.unescape(t)
        if u == t:
            break
        t = u
    t = strip_invisible(t)  # entities can decode to invisible chars
    t = _LITERAL_CRLF.sub(" ", t)
    junk = bool(_PDF_JUNK.search(t))
    if junk:
        t = _pdf_table_to_text(t)
    t = _IMG.sub("", t)
    t = _EOL_BACKSLASH.sub("", t)

    def link(m: re.Match) -> str:
        text = " ".join(m.group(1).split())
        if not text or (_FOOTNOTE.match(text) and "#" in m.group(2)):
            return ""  # empty anchors and footnote markers
        return f"[{text}]({m.group(2)})"

    t = _PCT_FOOTNOTE.sub(r"\1", _LINK.sub(link, t).replace("**", ""))
    lines = []
    for line in t.split("\n"):
        line = line.rstrip()
        if line.startswith("|"):
            lines.append(re.sub(r"[ \t]+", " ", line))
        else:
            lines.append(re.sub(r"[ \t]+", " ", _MD_ESCAPE.sub(r"\1", line)).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), junk


def line_key(line: str) -> str:
    """Comparable form of a line for frequency counting (links reduced to text, case/space folded)."""
    s = strip_md_links(line)
    return " ".join(re.sub(r"[*_`]+", "", s).lower().split())


def _protected(line: str, key: str) -> bool:
    return line.startswith("|") or "effective" in key or (bool(re.search(r"\d", key)) and bool(re.search(r"[%$]", key)))


class BoilerplateProfile:
    """Lines/headings that occur on more than `threshold` of a snapshot's pages."""

    def __init__(self, texts: list[str], threshold: float = 0.30, min_pages: int = 5):
        self.n_pages = len(texts)
        cut = max(min_pages, threshold * len(texts))
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update({line_key(l) for l in t.split("\n") if l.strip()})
        self.threshold = threshold
        self.keys = {k for k, c in counts.items() if c > cut and k}
        self.counts = {k: counts[k] for k in self.keys}
        self.heading_keys = {k.lstrip("# ") for k in self.keys if k.startswith("#")}

    def apply(self, text: str) -> str:
        """Drop boilerplate lines, UI chrome, boilerplate-headed sections, and adjacent duplicate lines."""
        segments: list[tuple[int, str, list[str]]] = []  # (level, heading line, body lines)
        level, head, body = 0, "", []
        for line in text.split("\n"):
            m = re.match(r"(#+)\s", line.strip())
            if m:
                segments.append((level, head, body))
                level, head, body = len(m.group(1)), line, []
            else:
                body.append(line)
        segments.append((level, head, body))

        out: list[str] = []
        skip_level = 0  # >0 while inside a dropped section
        prev = None
        for level, head, body in segments:
            if skip_level and level and level <= skip_level:
                skip_level = 0
            if head and not skip_level and line_key(head.strip()).lstrip("# ") in self.heading_keys:
                skip_level = level
                head, body = "", []  # own heading + direct body dropped
            elif skip_level:
                if not any(_protected(l.strip(), line_key(l.strip())) for l in body if l.strip()):
                    continue  # sub-section without data: dropped
            lines = ([head] if head else []) + body
            for line in lines:
                s = line.strip()
                key = line_key(s)
                if s and (
                    not key
                    or _UI_NOISE.match(key)
                    or _FLAG_ITEM.match(s)
                    or (key in self.keys and not _protected(s, key))
                ):
                    continue
                if s and not s.startswith("|") and s == prev:
                    continue
                if s:
                    prev = s
                out.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---- pages
RATE_SHEET, PRODUCT, BRANCH, NEWSROOM, BLOG, FORM_LEGAL, OTHER = (
    "rate_sheet",
    "product",
    "branch_contact",
    "newsroom",
    "blog",
    "form_legal",
    "other",
)

_FORM_PDF = re.compile(
    r"application|statement|pfs|privacy|enrol+ment|terms|conditions|release|authori[sz]ation|"
    r"layout|disclosure|notice",
    re.I,
)
_RATE_HEADER = re.compile(r"^\|.*\b(rate|apy|apr|dividend)\b.*\|", re.I | re.M)


@dataclass(frozen=True)
class PageClass:
    page_type: str
    fmt: str
    listing: bool


def classify(url: str, title: str, text: str, effective: list[str]) -> PageClass:
    """Page type from URL path (+ title/content for a few refinements)."""
    p = urlparse(url)
    path = p.path.rstrip("/").lower()
    ext = path.rsplit(".", 1)[-1] if re.search(r"\.[a-z0-9]{2,5}$", path) else ""
    fmt = ext if ext in ("pdf", "xlsx", "xls", "docx", "doc", "csv") else "html"
    if fmt != "html" and fmt != "pdf":
        return PageClass(OTHER, fmt, False)
    if fmt == "pdf":
        name = path.rsplit("/", 1)[-1]
        if "fee-schedule" in name:
            return PageClass(PRODUCT, fmt, False)
        return PageClass(FORM_LEGAL if _FORM_PDF.search(name) else OTHER, fmt, False)
    if path.startswith(LEGAL_PATHS):
        return PageClass(FORM_LEGAL, fmt, False)
    if "offer has expired" in title.lower():
        return PageClass(OTHER, fmt, False)
    if path in NEWSROOM_INDEX_PATHS or path.startswith(NEWSROOM_LISTING_PREFIXES):
        return PageClass(OTHER, fmt, True)
    if path.startswith(NEWSROOM_PREFIX):
        return PageClass(NEWSROOM, fmt, False)
    if path in BLOG_INDEX_PATHS or path.startswith(BLOG_LISTING_PREFIXES):
        return PageClass(OTHER, fmt, True)
    if path.startswith(BLOG_PREFIXES) or re.match(r"^/20\d\d/\d\d/", path):
        return PageClass(BLOG, fmt, False)
    if path in RATE_SHEET_PATHS:
        return PageClass(RATE_SHEET, fmt, False)
    if path in CONTACT_PATHS:
        return PageClass(BRANCH, fmt, False)
    if path == "":
        return PageClass(BRANCH if "lobby" in text.lower() else PRODUCT, fmt, False)
    if path.startswith(PRODUCT_PREFIXES) or path.endswith("-lp"):
        if effective and _RATE_HEADER.search(text):
            return PageClass(RATE_SHEET, fmt, False)
        return PageClass(PRODUCT, fmt, False)
    if re.search(r"/(?:news|newsroom|press|press-releases?|media)(?:/|$)", path):
        return PageClass(NEWSROOM, fmt, False)
    if re.search(r"/(?:blog|posts?|articles?|insights|stories|story)(?:/|$)", path):
        return PageClass(BLOG, fmt, False)
    return PageClass(OTHER, fmt, False)


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass(eq=False)
class ParsedPage:
    """Everything derived from one cleaned text; shared between snapshots when the content hash is unchanged."""

    text: str
    page_type: str
    fmt: str
    listing: bool
    junk_table: bool
    effective_dates: list[str]
    chunks: list[Chunk]


@dataclass(eq=False)
class PageVersion:
    """One page as seen in one snapshot."""

    url: str  # normalised (see urls.normalize_url)
    raw_url: str  # as in the manifest (what we cite)
    title: str
    snapshot_id: str
    taken_on: str
    fetched_at: str
    content_hash: str
    parsed: ParsedPage
    effective_date: str | None
    publish_date: str | None

    @property
    def page_type(self) -> str:
        return self.parsed.page_type


def make_version(
    raw_url: str, title: str, snapshot_id: str, taken_on: str, fetched_at: str, parsed: ParsedPage, chash: str
) -> PageVersion:
    url = normalize_url(raw_url)
    fetched_on = (fetched_at or taken_on)[:10]
    eff = pick_effective(parsed.effective_dates, max(fetched_on, taken_on)) if parsed.effective_dates else None
    pub = None
    if parsed.page_type in (NEWSROOM, BLOG):
        pub = publish_date(parsed.text, urlparse(raw_url).path)
    return PageVersion(url, raw_url.split("#")[0], title, snapshot_id, taken_on, fetched_at, chash, parsed, eff, pub)
