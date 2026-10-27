"""Text questions by extractive retrieval: fixed answers, search hits, scored sentences, then the refusal rules."""
from __future__ import annotations

import re
from functools import cached_property

from ..core.types import Result, SourceRef
from ..ingest.pages import FORM_LEGAL
from ..site_profile import INSTITUTION
from . import refuse
from .answer_type import has_expected_detail, has_figure
from .sentences import PageYears, render  # noqa: F401  (render: llm/read.py renders its passages with it)
from .text_compose import (
    SEARCH_K,
    SEARCH_PER_PAGE,
    Cand,
    CandidateScorer,
    Composer,
    QueryProfile,
)
from .text_shapes import (
    LIST_ASK,
    HubPages,
    count_answer,
    link_answer,
    list_answer,
    service_request,
    site_figure,
)
from .vocab import STOP_QUESTION_WORDS, question_terms, search_text, tokenize

BLANK_FORM = re.compile(r"application|statement|pfs|release", re.I)
OFFICIAL = re.compile(rf"\b{re.escape(INSTITUTION)}")


class TextAnswerer:
    def __init__(self, kb):
        self.kb, self.entities, self.years, self.hubs = kb, Entities(kb), PageYears(), HubPages(kb)
        self.composer = Composer(self.years)

    @staticmethod
    def _blank_form(page) -> bool:
        return (
            page.page_type == FORM_LEGAL
            and page.parsed.fmt == "pdf"
            and page.parsed.junk_table
            and bool(BLANK_FORM.search(page.url))
        )

    def _with_product_pages(self, q, view, hits, terms, search):
        """A product page whose URL names a question term ('student-loans' for 'student ..."""
        have, asked, extra = {h.page.url for h in hits}, set(terms) | set(tokenize(search)), []
        if len(asked) < 2:
            return hits
        for page in sorted(view.pages.values(), key=lambda p: p.url):
            if page.page_type != "product" or page.url in have or self._blank_form(page):
                continue
            slug = set(tokenize(re.sub(r"https?://[^/]+", "", page.url).replace("/", " ").replace("-", " ")))
            named = (slug & asked) - {"loan", "account", "personal", "business"}
            if not named or not any(view.index.idf.get(t, 0) > 2.5 for t in named) or len(slug & asked) < 2:
                continue
            found = [
                h
                for h in self.kb.search(" ".join(sorted(slug & asked)), q.as_of, k=SEARCH_K, per_page=SEARCH_PER_PAGE)
                if h.page.url == page.url
            ]
            if found:
                extra.append(
                    type(found[0])(page, hits[0].score * 0.9, found[0].chunks)
                )  # scored just under the best hit
        return hits + extra

    def candidates(self, q):
        """(hits, scored sentence candidates best first, terms) for a query; empty lists when nothing matches."""
        view = self.kb.view_for(q.as_of)
        if view is None:
            return [], [], []
        search = search_text(q.raw)
        hits = [
            h
            for h in self.kb.search(search, q.as_of, k=SEARCH_K, per_page=SEARCH_PER_PAGE)
            if not self._blank_form(h.page)
        ]
        if not hits:
            return [], [], []
        terms = question_terms(q.raw)
        hits = self._with_product_pages(q, view, hits, terms, search)
        profile = QueryProfile(q, hits, terms, view.index.idf, self.entities.of(q), search)
        return hits, CandidateScorer(profile, self.years).score(hits), profile.terms

    def reading_set(self, q, n: int = 5):
        try:
            cands = self.candidates(q)[1]
        except Exception:
            return [], 0.0
        out, seen = [], set()
        for c in cands:
            if (c.page.url, id(c.chunk)) not in seen:
                seen.add((c.page.url, id(c.chunk)))
                out.append(c)
                if len(out) >= n:
                    break
        return out, (cands[0].cov if cands else 0.0)

    def _fixed_answer(self, q) -> Result | None:
        """The answer (or refusal) of a question that needs no retrieval, or None."""
        refusal = self.kb.voice.refusal
        view = self.kb.view_for(q.as_of)
        if view is None:
            return refusal("as_of predates every snapshot")
        if not [t for t in question_terms(q.raw) if t.isalpha()]:
            return refusal("no content words in the question")
        stat = site_figure(self.kb, q)
        if stat is not None:
            return stat
        missing = (
            refuse.unknown_terms(q.raw, view.index.idf) if q.answerable != "yes" else []
        )  # 'yes': the LLM says evidence rules decide
        if missing:
            return refusal(f"the site never mentions {missing}")
        service = service_request(self.kb, q)
        return service if service is not None else count_answer(self.kb, q)

    def answer(self, q, strict_numeric: bool = False) -> Result:
        """Extractive answer, or a no-number refusal."""
        refusal = self.kb.voice.refusal
        fixed = self._fixed_answer(q)
        if fixed is not None:
            return fixed
        hits, cands, terms = self.candidates(q)
        if not hits:
            return refusal("no search hits")
        linked = link_answer(q, hits, terms) if q.intent == "url" else None
        if linked:
            return linked
        if not cands:
            return refusal("no sentence covers the question")
        if strict_numeric:
            # a genuine rate / fee ask: the answer sentence carries the figure (an article's must name the institution)
            top = cands[0].score
            cands = [
                c
                for c in cands
                if has_figure(q, c.sent)
                and c.score >= 0.3 * top
                and (c.page.page_type not in ("blog", "newsroom") or OFFICIAL.search(c.sent))
            ]
            if not cands:
                return refusal("no figure in the retrieved passages")
        listed = list_answer(self.kb, self.hubs, q, hits) if LIST_ASK.search(q.raw) else None
        if listed is not None:
            return listed
        return self._extract(q, hits, cands, terms, strict_numeric)

    def _extract(self, q, hits, cands, terms, strict_numeric: bool) -> Result:
        """The best sentence composed into text, unless a refusal rule says it does not answer the question."""
        refusal = self.kb.voice.refusal
        best = self.composer.newest_near_duplicate(cands[0], cands, q)
        idf = self.kb.view_for(q.as_of).index.idf
        unknown = refuse.unknown_entity(q.raw, idf)
        ents = self.entities.of(q)
        why = f"question names '{unknown}', which appears nowhere on the site" if unknown else None
        why = why or (refuse.entity_gap(q, ents, best, hits) if ents else None)
        if why:
            return refusal(why)
        varied = self.composer.variants(q, cands)
        if varied is not None:
            return varied
        if best.cov < (0.5 if strict_numeric else 0.28) and not refuse.supported_by_chunk(q, best, terms):
            return refusal(f"best sentence covers only {best.cov:.2f} of the question's terms")
        text = self.composer.compose(q, best, cands)
        if refuse.link_only(text):
            why = "the best match is only a link to another organisation"
        elif refuse.general_interest_only(q, best, text):
            why = f"only general-interest text, nothing that says {INSTITUTION} offers it"
        elif not has_expected_detail(q, text):
            why = "the passage lacks the kind of detail asked for"
        elif not refuse.is_sufficient(q, text, best, terms, idf, strict_numeric):
            why = "the passage covers too little of the question"
        return refusal(why) if why else self._sourced(best, cands, text)

    @staticmethod
    def _sourced(best: Cand, cands: list[Cand], text: str) -> Result:
        srcs, seen, words = (
            [SourceRef(best.page.raw_url, best.page.snapshot_id, best.sent[:200])],
            {best.page.url},
            set(tokenize(text)),
        )
        for c in cands[:12]:
            ctoks = set(tokenize(c.sent))
            if (
                c.page.url not in seen
                and len(srcs) < 3
                and ctoks
                and c.score >= 0.6 * best.score
                and len(ctoks & words) / len(ctoks) >= 0.7
            ):
                seen.add(c.page.url)
                srcs.append(SourceRef(c.page.raw_url, c.page.snapshot_id, c.sent[:200]))
        return Result(text, srcs, True, None, extractive=True)

# ---- Proper names in a question that are not a city or street of the site (a person, place or program).
_ADDRESS_LINE = re.compile(r"(?m)^\s*(\d{1,6} [^,\n]{3,40}), ([A-Za-z .'-]{3,30}), ([A-Z]{2}),? \d{5}")
# street words and names that are not distinctive enough to identify a place
_PLAIN_STREET_WORDS = {
    "street",
    "avenue",
    "road",
    "drive",
    "lane",
    "boulevard",
    "center",
    "first",
    "second",
    "north",
    "south",
    "east",
    "west",
    "main",
    "highway",
    "park",
    "john",
    "kennedy",
}
# capitalised words in a question that name nothing
_NOT_ENTITIES = {
    INSTITUTION,
    "I",
    "Shine",
    "Visa",
    "CD",
    "IRA",
    "APR",
    "APY",
    "ATM",
    "Avenue",
    "Street",
    "Road",
    "Drive",
    "Lane",
    "Boulevard",
    "Branch",
    "Location",
    "Online",
    "Banking",
    "Mobile",
}


class Entities:
    """Finds the names a question uses; knows the cities and streets of the latest snapshot's published addresses."""

    def __init__(self, kb):
        self.kb = kb

    @cached_property
    def gazetteer(self) -> set:
        """City and street-name tokens of every street address published on the latest snapshot (branches, ATMs)."""
        g: set = set()
        for page in (self.kb.views[-1].pages.values() if self.kb.views else ()):
            for m in _ADDRESS_LINE.finditer(page.parsed.text or ""):
                g |= {t for t in tokenize(m.group(2)) if len(t) >= 4}
                g |= {
                    t
                    for t in tokenize(re.sub(r"^\d+\s*", "", m.group(1)))
                    if len(t) >= 5 and t not in _PLAIN_STREET_WORDS
                }
        return g

    def of(self, q) -> set:
        """Tokens naming something in the question."""
        words = re.findall(r"(?<=[a-z,] )([A-Z][\w$-]*)", q.raw)
        # all-capital words up to this long are acronyms, not names
        names = " ".join(w for w in words if w not in _NOT_ENTITIES and not (w.isupper() and len(w) <= 5))
        ents = set(tokenize(names))
        ents |= {t for t in tokenize(q.raw) if t in self.gazetteer and t not in STOP_QUESTION_WORDS}
        return ents

