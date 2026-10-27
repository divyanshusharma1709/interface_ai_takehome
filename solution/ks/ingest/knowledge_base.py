"""The knowledge base: per-snapshot page views, fact extraction across a snapshot, and search."""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass, field

from ..core.dates import effective_dates, parse_as_of
from ..core.urls import normalize_url
from ..site_profile import DEFAULT_VOICE, voice_for
from .chunks import JUNK_TABLE_WEIGHT, LISTING_WEIGHT, TYPE_WEIGHT, BM25Index, Chunk, Doc, chunk_page
from .facts import Conf, ExtractionStats, Fact, ProductRegistry, make_key
from .line_facts import extract_fee_lines, extract_lines
from .normalize import strip_segment, tokens
from .pages import BoilerplateProfile, PageVersion, ParsedPage, classify, content_hash, make_version, precleaned
from .prose_facts import extract_prose
from .store import FactStore
from .table_facts import extract_tables, is_fee_schedule_pdf


# ---- page_facts
def extract_page(page, stats) -> list[Fact]:
    """Table rows, plus the fee lines when the page is a fee-schedule PDF."""
    out = extract_tables(page, stats)
    if is_fee_schedule_pdf(page):
        out.extend(extract_fee_lines(page, stats))
    elif page.parsed.junk_table:
        stats["junk_pages_skipped"] += 1
    return out


# ---- snapshot_facts
OCR_NAME_SIMILARITY = 0.9
YIELD_TOLERANCE = 1e-9


class SnapshotExtractor:
    def __init__(self) -> None:
        self.stats = ExtractionStats()
        self.errors: list[str] = []
        self.registry = ProductRegistry()
        self.named = ProductRegistry()  # products named by their own table rows
        self._templates: dict = {}

    def extract(self, view) -> list[Fact]:
        """The observations of one `SnapshotView`."""
        facts = self._table_facts(view)
        for f in facts:
            if f.source_kind == "table" and "label_only" not in f.flags and f.confidence >= Conf.ANSWER_MIN:
                self.named.add(f.product, f.family)
        self._resolve_labels(facts)
        self._flag_garbled(facts)
        self._flag_inverted(facts)
        for f in facts:
            if f.source_kind == "table" and f.confidence >= Conf.ANSWER_MIN:
                self.registry.add(f.product, f.family)
        known_fams = {
            f.family
            for f in facts
            if f.source_kind in ("table", "fee_line") and f.family and f.confidence >= Conf.ANSWER_MIN
        }
        known_fams |= self.registry.families()
        return facts + self._text_facts(view, known_fams)

    # ---- table and fee-line facts
    def _table_facts(self, view) -> list[Fact]:
        out: list[Fact] = []
        for url, page in sorted(view.pages.items()):
            for obs in self._page_observations(page):
                f = obs.placed(
                    url=page.url, snapshot_id=view.snapshot_id, taken_on=view.taken_on, fetched_at=page.fetched_at
                )
                f.raw_url = f.raw_url or page.raw_url
                if f.effective_date is None and page.effective_date and f.source_kind != "prose":
                    f.effective_date, f.effective_scope = page.effective_date, "page"
                out.append(f)
        return out

    def _page_observations(self, page) -> list[Fact]:
        if page.parsed not in self._templates:
            try:
                self._templates[page.parsed] = extract_page(page, self.stats)
            except Exception as e:
                self.errors.append(f"{page.url}: {type(e).__name__}: {e}")
                self._templates[page.parsed] = []
        return self._templates[page.parsed]

    # ---- prose and line facts
    def _text_facts(self, view, known_fams: set) -> list[Fact]:
        out: list[Fact] = []
        for url, page in sorted(view.pages.items()):
            try:
                found = extract_prose(page, self.registry, self.stats) + extract_lines(
                    page, self.registry, known_fams, self.stats
                )
            except Exception as e:
                self.errors.append(f"{url} (prose): {type(e).__name__}: {e}")
                found = []
            for obs in found:
                f = obs.placed(
                    url=page.url, snapshot_id=view.snapshot_id, taken_on=view.taken_on, fetched_at=page.fetched_at
                )
                f.raw_url = f.raw_url or page.raw_url
                out.append(f)
        return out

    # ---- repairs across facts
    @staticmethod
    def _flag_garbled(facts: list[Fact]) -> None:
        groups: dict[tuple, dict[str, list[Fact]]] = defaultdict(lambda: defaultdict(list))
        for f in facts:
            if f.source_kind == "fee_line":
                groups[(f.value, f.key.qualifiers)][f.product].append(f)
        for names in groups.values():
            for a in names:
                for b in names:
                    if (
                        a != b
                        and len(b.split()) > len(a.split())
                        and difflib.SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()
                        >= OCR_NAME_SIMILARITY
                    ):
                        for f in names[b]:
                            f.confidence = min(f.confidence, Conf.SUSPECT)
                            f.flags = tuple(sorted(set(f.flags) | {"ocr_suspect"}))

    @staticmethod
    def _flag_inverted(facts: list[Fact]) -> None:
        rows: dict[tuple, dict[str, Fact]] = defaultdict(dict)
        for f in facts:
            if f.source_kind == "table" and f.attribute in ("apy", "dividend_rate") and f.value is not None:
                rows[(f.url, f.product, f.key.qualifiers, f.table_index)][f.attribute] = f
        for row in rows.values():
            a, d = row.get("apy"), row.get("dividend_rate")
            if a and d and a.value < d.value - YIELD_TOLERANCE:
                d.confidence = min(d.confidence, Conf.SUSPECT)
                d.flags = tuple(sorted(set(d.flags) | {"inverted_yield"}))

    def _resolve_labels(self, facts: list[Fact]) -> None:
        for f in facts:
            if f.source_kind != "table" or "label_only" not in f.flags or f.product in self.named:
                continue
            caption_words = frozenset(tokens(strip_segment(f.caption)[0])) - {"rate", "account"}
            fuller = sorted(p for p, _, toks in self.named.entries() if caption_words and caption_words < toks)
            if len(fuller) == 1:
                p = fuller[0]
                f.key = make_key(p, f.attribute, f.key.qdict)
                f.family = self.named.family(p)
                f.flags = tuple(sorted(set(f.flags) | {"label_resolved"}))


# ---- knowledge_base
@dataclass
class SnapshotView:
    """The website as of one snapshot: page versions plus a BM25 index over their chunks."""

    snapshot_id: str
    taken_on: str
    pages: dict[str, PageVersion]
    index: BM25Index
    doc_pages: list[PageVersion] = field(default_factory=list)  # parallel to index.docs


@dataclass
class Hit:
    """Best-matching chunks of one page in one snapshot view."""

    page: PageVersion
    score: float
    chunks: list[tuple[Chunk, float]]  # top chunks of this page, best first

    @property
    def best(self) -> Chunk:
        return self.chunks[0][0]


class KnowledgeBase:
    """Ingest snapshots oldest->newest; search any snapshot view by `as_of` date."""

    def __init__(self, boilerplate_threshold: float = 0.30):
        self.threshold = boilerplate_threshold
        self.views: list[SnapshotView] = []
        self.voice = DEFAULT_VOICE
        self.profiles: dict[str, BoilerplateProfile] = {}
        self.extractor = SnapshotExtractor()
        self.facts = FactStore()  # M2: bi-temporal numeric facts
        self._pre_cache: dict[int, tuple[str, bool]] = {}  # hash(raw markdown) -> precleaned
        self._parse_cache: dict[tuple[str, str, str], ParsedPage] = {}  # (url, title, content hash) -> parse

    # ------------------------------------------------------------------ ingest
    def ingest(self, snapshot) -> SnapshotView:
        """Ingest one snapshot."""
        pre: dict[str, tuple[str, bool]] = {}
        for p in snapshot.pages:
            key = hash(p.markdown)
            if key not in self._pre_cache:
                self._pre_cache[key] = precleaned(p.markdown)
            pre[p.url] = self._pre_cache[key]
        profile = BoilerplateProfile([t for t, _ in pre.values()], self.threshold)
        self.profiles[snapshot.snapshot_id] = profile

        groups: dict[str, list] = {}
        for p in snapshot.pages:
            groups.setdefault(normalize_url(p.url), []).append(p)
        pages: dict[str, PageVersion] = {}
        for nurl, variants in groups.items():
            parts = []
            for p in variants:
                text = profile.apply(pre[p.url][0])
                parts.append((p, text, pre[p.url][1], content_hash(text)))
            parts.sort(key=lambda t: (t[0].fetched_at, t[0].url), reverse=True)  # newest crawl first = primary
            p0 = parts[0][0]
            chash = parts[0][3] if len(parts) == 1 else content_hash("\x00".join(sorted(t[3] for t in parts)))
            ck = (nurl, p0.title, chash)
            parsed = self._parse_cache.get(ck)
            if parsed is None:
                parsed = self._parse_variants(nurl, p0.title, parts)
                self._parse_cache[ck] = parsed
            pages[nurl] = make_version(
                p0.url, p0.title, snapshot.snapshot_id, snapshot.taken_on, p0.fetched_at, parsed, chash
            )

        self.voice = voice_for([p.url for p in snapshot.pages])
        self.views = [v for v in self.views if v.snapshot_id != snapshot.snapshot_id]
        docs, doc_pages = [], []
        for v in pages.values():
            w = (
                TYPE_WEIGHT[v.page_type]
                * (LISTING_WEIGHT if v.parsed.listing else 1.0)
                * (JUNK_TABLE_WEIGHT if v.parsed.junk_table else 1.0)
            )
            for c in v.parsed.chunks:
                docs.append(Doc(c, w))
                doc_pages.append(v)
        view = SnapshotView(snapshot.snapshot_id, snapshot.taken_on, pages, BM25Index(docs), doc_pages)
        self.views.append(view)
        self.views.sort(key=lambda s: s.taken_on)
        self.facts.add_snapshot(view, self.extractor.extract(view))
        return view

    def _parse_variants(self, nurl: str, title: str, parts: list) -> ParsedPage:
        eff: list[str] = []
        chunks: list = []
        seen: set[str] = set()
        primary = None
        for p, text, junk, _ in parts:
            e = effective_dates(text)
            eff = sorted(set(eff) | set(e))
            pc = classify(p.url, p.title, text, e)
            ch = chunk_page(nurl, p.title, pc.page_type, text, p.url)
            if primary is None:
                primary = (text, pc, junk)
            for c in ch:
                h = content_hash(c.text)
                if h not in seen:
                    seen.add(h)
                    chunks.append(c)
        for i, c in enumerate(chunks):
            c.ordinal = i
        text, pc, junk = primary
        for c in chunks:
            c.page_type = pc.page_type
        return ParsedPage(text, pc.page_type, pc.fmt, pc.listing, junk, eff, chunks)

    # ------------------------------------------------------------------ views / records
    def view_for(self, as_of: str | None) -> SnapshotView | None:
        """Latest snapshot taken on or before `as_of`; None if `as_of` predates every snapshot."""
        if not self.views:
            return None
        d = parse_as_of(as_of)
        if d is None:
            return self.views[-1]
        ok = [v for v in self.views if v.taken_on <= d]
        return ok[-1] if ok else None

    # ------------------------------------------------------------------ search
    def search(self, query: str, as_of: str | None = None, k: int = 10, per_page: int = 3) -> list[Hit]:
        """Top-k pages (one group per URL) by weighted BM25 in the snapshot view for `as_of`."""
        view = self.view_for(as_of)
        if view is None:
            return []
        scores = view.index.scores(query)
        by_url: dict[str, list[tuple[float, int]]] = {}
        for i, s in scores.items():
            by_url.setdefault(view.doc_pages[i].url, []).append((s, i))
        hits = []
        for url, lst in by_url.items():
            lst.sort(key=lambda t: (-t[0], t[1]))
            page = view.pages[url]
            hits.append(Hit(page, lst[0][0], [(view.index.docs[i].chunk, s) for s, i in lst[:per_page]]))
        hits.sort(key=lambda h: (-h.score, h.page.url))
        return hits[:k]
