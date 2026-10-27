"""Scores the sentences of the search hits against a question, then turns the best one into answer text."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.textutil import NUMBER_TENS, NUMBER_UNITS
from ..core.types import Result, SourceRef
from ..site_profile import INSTITUTION, path_only, strip_title_suffix
from .answer_type import (
    HOURS_RX,
    NUMBER_WORDS,
    TypeBoost,
    amount_predicate,
    expected_detail,
)
from .sentences import (
    PageYears,
    is_teaser,
    page_authority,
    split_sentences,
    split_with_context,
)
from .vocab import SCORE_SYNONYMS, STOP_QUESTION_WORDS, question_terms, tokenize

# ---- Scores the sentences of the search hits against a question.
SEARCH_K, SEARCH_PER_PAGE = 14, 5
MONTHS = "january february march april may june july august september october november december".split()
RECENCY_ASK = re.compile(r"\b(?:latest|most recent|newest|last round|current round|this year'?s?)\b", re.I)
HOW_MANY_NOUN = re.compile(
    r"\bhow many\s+(?:(?:did|do|does|have|has|were|are|is|was|of|the|total|different|more|other)\s+)*([a-z]{4,})"
)
JOIN_VERB = re.compile(r"\b(?:join|open|become|start)\w*", re.I)
COINED_NAME = re.compile(r"\b[A-Z][a-z]+(?: (?:of|the|and|for) [A-Z][a-z]+| [A-Z][a-z]+)+\b")
QUESTION_START = re.compile(r"(?:What|How|Who|When|Where|Why|Which|Does|Do|Is|Are|Can)\b")
STAT_WORD = re.compile(r"\b(average|median|total|combined)\b", re.I)
RECAP = re.compile(
    r"\b(?:inaugural|previous\w*|prior|past|earlier|formerly|originally|(?:first|second|third) (?:year|round|cycle))\b",
    re.I,
)
URL, YEAR, DOLLAR = re.compile(r"https?://\S+"), re.compile(r"\b(?:19|20)\d\d\b"), re.compile(r"\$\s?\d")
SLUG_NOISE = {"personal", "business", INSTITUTION.lower()}


@dataclass
class Cand:
    sent: str
    cov: float
    score: float
    page: object
    chunk: object
    pos: int
    pub: str
    group: int = -1            # id of the bullet list the sentence belongs to (-1 = not a bullet)
    teaser: bool = False       # another article's card on this page


class QueryProfile:
    """What scoring needs to know about the question, computed once."""

    def __init__(self, q, hits, terms: list, idf: dict, entities: set, search: str):
        raw, self.q, self.entities = q.raw, q, entities
        for rx, extra in SCORE_SYNONYMS:
            if rx.search(raw):
                terms += [t for t in tokenize(extra) if t not in terms]
        self.recency = RECENCY_ASK.search(raw)
        if self.recency:           # 'latest' / 'this year' say which release to prefer, not what to find in it
            drop = {
                t
                for t in tokenize(self.recency.group(0))
                if t in ("latest", "recent", "newest", "year", "last", "current")
            }
            terms = [t for t in terms if t not in drop]
        self.terms, self.idf = terms, idf
        self.dates = sorted({h.page.publish_date for h in hits if h.page.publish_date})
        self.need, self.predicate, self.boost = expected_detail(q), amount_predicate(q), TypeBoost(q)
        cm = HOW_MANY_NOUN.search(raw.lower())
        head = r"(?<![$\d,.])\b(?:\d[\d,]*|" + NUMBER_WORDS + r")\s+(?:[\w’'-]+\s+){0,"
        noun = re.escape(cm.group(1)[:5]) + r"\w*" if cm else ""
        self.counted, self.counted_near = (
            (re.compile(head + n + "}" + noun, re.I) for n in ("5", "1")) if cm else (None, None)
        )
        self.verb = JOIN_VERB if self.predicate is not None and JOIN_VERB.search(raw) else None
        self.coined = [m.group(0).lower() for m in COINED_NAME.finditer(raw) if not QUESTION_START.match(m.group(0))]
        self.stat_word = STAT_WORD.search(raw)
        self.stat_rx = (
            re.compile(r"\b" + self.stat_word.group(1).lower() + r"\b|\b(?:in all|altogether|overall)\b", re.I)
            if self.stat_word
            else None
        )
        self.month = MONTHS[q.month_hint - 1] if q.month_hint else None
        self.llm_entities = set(tokenize(" ".join(q.entities)))
        ordered = [t for t in tokenize(re.sub(r"[?!.,;:]", " ", raw)) if t not in STOP_QUESTION_WORDS]
        self.pairs = set(zip(ordered, ordered[1:]))
        top_idf = max(idf.values()) if idf else 1.0
        self.weights = {t: idf.get(t, top_idf) for t in terms}
        self.total = sum(self.weights.values()) or 1.0
        self.expanded = set(terms) | set(tokenize(search))
        self.newest_wanted = bool(self.recency) and not q.year_hint

    def wsum(self, tokens: set, skip: set = frozenset()) -> float:
        """Weight of the question's terms that are in `tokens` and not in `skip`."""
        return sum(self.weights[t] for t in self.terms if t in tokens and t not in skip)


class CandidateScorer:
    """Scores every sentence of the retrieved chunks (`score(hits)`), best first."""

    def __init__(self, profile: QueryProfile, years: PageYears):
        self.p, self.q, self.years = profile, profile.q, years

    def score(self, hits) -> list[Cand]:
        """All scored sentence candidates of `hits`, best first (ties: page URL, position)."""
        p = self.p
        best = hits[0].score or 1.0
        if self.q.year_hint:        # relative to the best release of the asked year, not to a year-less program page
            same_year = [h.score for h in hits if self.years.of(h.page) == self.q.year_hint]
            best = max(same_year) or best if same_year else best
        cands = [c for h in hits for chunk, cs in h.chunks for c in self._chunk(h, chunk, cs, best)]
        if p.newest_wanted and cands:    # among the pages that speak to it, the newest release's sentences win
            top = max(c.score for c in cands)
            near = [c for c in cands if c.score >= 0.3 * top and c.pub and not c.teaser]
            newest = max((c.pub for c in near), default="")
            for c in near:
                if c.pub == newest:
                    c.score *= 1.8
        cands.sort(key=lambda c: (-c.score, c.page.url, c.pos))
        return cands

    def _chunk(self, h, chunk, cs: float, best: float) -> list[Cand]:
        p, page = self.p, h.page
        title = set(tokenize(page.title or ""))
        slug = set(tokenize(re.sub(r"https?://[^/]+", "", page.url).replace("/", " ").replace("-", " ")))
        core = slug - SLUG_NOISE
        named = core & p.expanded     # a product page whose URL names the question is authoritative for it
        url_share = (
            len(named) / len(core)
            if page.page_type == "product" and len(named) >= 2 and sum(p.idf.get(t, 0.0) for t in named) >= 4.0
            else 0.0
        )
        recency = (
            p.dates.index(page.publish_date) / (len(p.dates) - 1) if len(p.dates) > 1 and page.publish_date else 0.0
        )
        # chunk relevance, hit relevance, page authority and how much the page is about the question: multiplied in this
        # order
        rel, hit_rel = 0.5 + 0.5 * min(1.0, cs / best), 0.6 + 0.4 * min(1.0, h.score / best)
        authority, about = page_authority(page, self.q), 1.0 + 0.8 * (p.wsum(title) / p.total)
        chunk_cov = p.wsum(set(tokenize(chunk.text)) | set(tokenize(chunk.heading_path))) / p.total
        pub, out = page.publish_date or "", []
        for pos, (s, grp, ctx) in enumerate(split_with_context(chunk.text)):
            st = set(tokenize(s))
            cov = p.wsum(st) / p.total + 0.5 * p.wsum(title, st) / p.total      # the page names the entity
            cov += 0.3 * max(0.0, chunk_cov - cov)                    # its passage is about it
            head = 0.0
            ctx = ctx or (chunk.heading_path.split(" > ")[-1] if grp >= 0 else "")
            if ctx:                                                   # a bullet inherits its heading
                ctoks = set(tokenize(ctx + " " + " ".join(chunk.heading_path.split(" > ")[1:])))
                cov += 0.7 * p.wsum(ctoks, st | title) / p.total
                hc = 0.9 * p.wsum(ctoks) / p.total
                if hc > cov and hc >= 0.55:
                    cov, head = min(1.0, hc + 0.1), hc                # the heading answers it
            if cov <= 0:
                continue
            length = (
                0.6
                if len(re.sub(r"\(?https?://\S+", "", s).split()) <= 4 and not re.search(r"\d", s)
                else (1.0 if len(s) <= 320 else 0.85)
            )
            score = (cov + self._boost(s, title)) * rel * hit_rel * length * authority * about
            for f in self._factors(s, st, chunk, page, title, head, recency, url_share, pub):
                score *= f
            teaser = is_teaser(page.parsed.text or "", s, page.url)
            if teaser and p.recency:    # another article's card: the article's own page speaks for it
                score *= 0.4
            out.append(Cand(s, cov, score, page, chunk, pos, pub, grp, teaser))
        return out

    def _boost(self, s: str, title: set) -> float:
        p = self.p
        boost = p.boost.score(s)
        if p.pairs:
            seq = tokenize(s)
            pairs = {pr for pr in p.pairs if not (pr[0] in title and pr[1] in title)}      # not the page's own name
            boost += min(0.6, 0.3 * len(pairs & set(zip(seq, seq[1:]))))                   # 'annual fee' as a phrase
        return boost

    def _factors(self, s, st, chunk, page, title, head, recency, url_share, pub):
        """The multipliers of one sentence's score, in order (a signal that does not apply contributes none)."""
        p, q = self.p, self.q
        yield (1.08 if chunk.ordinal == 0 else 1.0) * (
            1.0 + (0.6 if p.newest_wanted else 0.12) * recency
        )  # leads summarise; newer wins ties
        if p.entities:
            yield 0.45 + 0.55 * sum(1 for e in p.entities if e in st | title) / len(
                p.entities
            )  # named things must appear
        if head:
            yield 1.0 + 1.2 * head  # a list under a matching heading
        if s.rstrip().endswith("?"):
            yield 0.5  # an FAQ heading, not its answer
        if p.need is not None and not p.need.search(URL.sub("", s)):
            yield 0.35  # not the kind of detail asked for
        if p.predicate is not None and DOLLAR.search(s):  # an amount tied to the asked predicate
            yield 1.6 if p.predicate.search(s) else 0.6
            if p.verb is not None and p.predicate.search(s) and p.verb.search(s):
                yield 1.4
        if p.counted is not None:  # 'how many employees' -> a number right in front of the noun
            yield (1.9 if p.counted_near.search(s) else 1.5) if p.counted.search(s) else 0.8
        if q.year_hint:  # a year-specific fact: prefer the sentence / article of that year
            yield from self._year_factors(s, page, pub)
        if p.newest_wanted and RECAP.search(s):
            yield 0.4  # a recap of earlier rounds, not the latest one
        if p.coined:                                                           # a coined program name as a phrase
            wide = chunk.text.lower()[:1500] + " " + (page.title or "").lower()
            yield 1.6 if any(c in s.lower() for c in p.coined) else (1.15 if any(c in wide for c in p.coined) else 0.75)
        if p.stat_rx is not None:  # the asked statistic type (average / total)
            hit = bool(p.stat_rx.search(s))
            yield (
                (4.0 if hit else 0.4)
                if p.stat_word.group(1).lower() in ("average", "median")
                else (2.0 if hit else 0.8)
            )
        yield 1.0 + 1.0 * url_share
        if p.llm_entities and p.llm_entities & set(tokenize(s)):
            yield 1.25  # a name the LLM parser found in the question

    def _year_factors(self, s: str, page, pub: str):
        q, p = self.q, self.p
        yrs = set(YEAR.findall(s))
        if str(q.year_hint) in yrs:
            yield 1.6
        elif yrs:
            yield 0.4  # a sentence about another year (a recap in a later release)
        ref = self.years.of(page)
        if ref is not None:  # the year the release is about; an annual review is published early the next year
            late = ref == q.year_hint + 1 and not yrs and pub[5:7] in ("01", "02", "03")
            yield 1.9 if ref == q.year_hint else (1.15 if late else 0.5)
        if p.month and (p.month in s.lower() or p.month[:3] + "." in s.lower() or pub[5:7] == "%02d" % q.month_hint):
            yield 1.15

# ---- Turns the best sentence into answer text: companions, FAQ answers, figure wording and variants.
_URL = re.compile(r"https?://\S+")
_HOW_MANY = re.compile(r"\bhow many\b", re.I)
_AMOUNT_ASK = re.compile(r"\s*how (?:many|much)\b", re.I)
_DEFINITION_ASK = re.compile(r"\s*(?:what(?:'s| is| are)|tell me about|explain|describe)\b", re.I)
_REQUIREMENTS_ASK = re.compile(
    r"\b(?:eligib\w*|requirements?|qualif\w*|criteria|conditions?|who can|what (?:do|does) (?:i|it) need)\b", re.I
)
_AREA_WORDS = re.compile(r"\b(?:area|region|counties|county|territory|service area)\b", re.I)
_PLACE_LIST = re.compile(r"\b(?:in|across|throughout)\s+[A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*,?\s+and\s+[A-Z][a-z]+")
_CALL_TO_ACTION = re.compile(r"\s*(?:apply|learn more|get started|open|join|log in|sign|start|see|view|click)\b", re.I)
_COMPANION_STAT = re.compile(r"\W*(?:on average|average|median|total|combined|overall|in all|in total)\b", re.I)
_SAME_OR_DIFFERENT = re.compile(r"\b(?:same|differ\w*|vary|varies|varied|uniform|consistent)\b", re.I)
_HOURS_WORD = re.compile(r"\bhours?\b", re.I)

# lengths (characters)
# an amount sentence longer than this is cut at a word
_AMOUNT_MAX = 480
_LIST_CAP, _REQUIREMENTS_CAP, _LONG_LIST_CAP, _DEFAULT_CAP = 5, 8, 760, 480
_SAME_CHUNK_SHARE, _SAME_PAGE_SHARE = 0.5, 0.85
_NEAR_DUPLICATE_SIM, _NEAR_DUPLICATE_SHARE = 0.45, 0.5

_UNITS, _TENS = NUMBER_UNITS, NUMBER_TENS
_NUMBER_WORD = re.compile(
    r"\b(?:(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(list(_UNITS)[:9]) + r"))?|(" + "|".join(_UNITS) + r"))\b",
    re.I,
)


def with_digits(text: str) -> str:
    def sub(m: re.Match) -> str:
        tens, unit, small = (g.lower() if g else None for g in m.groups())
        n = _UNITS[small] if small else _TENS[tens] + (_UNITS[unit] if unit else 0)
        tail = text[m.end():m.end() + 6]
        if re.match(r"\s*\(\d|-", tail) or str(n) in text[max(0, m.start() - 8):m.start()]:
            return m.group(0)
        return f"{m.group(0)} ({n})"
    # a range or several counts ('three to six months'): leave the wording as written
    if len(_NUMBER_WORD.findall(text)) != 1:
        return text
    return _NUMBER_WORD.sub(sub, text)


def similarity(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / max(1, len(x | y))


def page_lead(page) -> str:
    """The page title plus its first two prose sentences (no links, headings or buttons)."""
    out = []
    for raw in (page.parsed.text or "").split("\n"):
        line = raw.strip()
        if len(line) < 40 or line[0] in "#[-|*>" or line.startswith("!") or "](" in line:
            continue
        out.extend(x for x, _ in split_sentences(line))
        if len(out) >= 2:
            break
    title = strip_title_suffix(page.title or "").strip()
    return (title + ": " if title and out else "") + " ".join(out[:2])


class Composer:
    """Composes answer text; `years` says which year a dated release is about."""

    def __init__(self, years):
        self.years = years
        self._page_tokens: dict = {}

    # ------------------------------------------------------------------ the answer text
    def compose(self, q, best: Cand, cands: list[Cand]) -> str:
        out = self._compose(q, best, cands)
        return with_digits(out) if _HOW_MANY.search(q.raw) else out

    def _compose(self, q, best: Cand, cands: list[Cand]) -> str:
        need = expected_detail(q)
        if (
            _AMOUNT_ASK.match(q.raw)
            and not best.sent.rstrip().endswith("?")
            and need is not None
            and need.search(_URL.sub("", best.sent))
        ):
            return self._amount(q, best, cands, need)
        faq = self._faq_answer(best)
        if faq is not None:
            return faq
        if best.page.page_type == "product" and _CALL_TO_ACTION.match(best.sent):
            # the winner is a button or an FAQ heading: the page's own description is the answer
            lead = page_lead(best.page)
            if lead:
                return lead
        if best.group >= 0:
            sents = self._list_items(q, best)
        elif _DEFINITION_ASK.match(q.raw):
            sents = self._definition(best)
        else:
            sents = self._page_companions(best, cands)
        return self._finish(q, best, sents)

    def _amount(self, q, best: Cand, cands: list[Cand], need) -> str:
        """The amount sentence, not its neighbours -- except parallel items and a statistic's companion."""
        if len(best.sent) > _AMOUNT_MAX:
            return best.sent[:_AMOUNT_MAX - 3].rsplit(" ", 1)[0] + "..."
        out = best.sent.strip()
        flat = [x for x, _ in split_sentences(best.chunk.text)]
        asked = set(question_terms(q.raw))
        overlap = lambda x: asked & set(tokenize(x))
        ties = [
            c
            for c in cands
            if c.chunk is best.chunk
            and c.group == best.group
            and c.sent != best.sent
            and not c.teaser
            # shares of the best score a companion must reach
            # items the wording cannot tell apart
            and c.score >= 0.97 * best.score
            and overlap(c.sent) == overlap(best.sent)
            and need.search(_URL.sub("", c.sent))
        ]
        # parallel items that the wording cannot tell apart (Ten $4,000 ... / Ten $2,000 ...): all of them are the
        # answer
        if best.group >= 0 and ties:
            items = sorted(
                {best.sent: 0, **{c.sent: 1 for c in ties}}, key=lambda x: flat.index(x) if x in flat else 99
            )
            joined = " ".join(x.strip() for x in items[:4])
            # the parallel items together
            return joined if len(joined) <= 560 else out
        # a statistic and its companion in the next sentence (a total, then the average) read the same whichever the
        # wording asked
        if best.group < 0 and best.sent in flat:
            at = flat.index(best.sent)
            for nxt in flat[at + 1:at + 3]:
                if (
                    re.search(r"\d", nxt)
                    and _COMPANION_STAT.match(nxt)
                    and not nxt.rstrip().endswith("?")
                    # a statistic and its companion sentences together
                    and len(out) + len(nxt) + 1 <= 420
                ):
                    out += " " + nxt.strip()
                else:
                    break
        return out

    @staticmethod
    def _faq_answer(best: Cand) -> str | None:
        """An FAQ question: its answer is what follows it."""
        if not best.sent.rstrip().endswith("?"):
            return None
        flat = [x for x, _ in split_sentences(best.chunk.text)]
        if best.sent not in flat:
            return None
        at = flat.index(best.sent)
        answer = [x for x in flat[at + 1:at + 3] if not x.rstrip().endswith("?")]
        return " ".join([best.sent] + answer)[:480] if answer else None

    @staticmethod
    def _list_items(q, best: Cand) -> list[str]:
        """The sentences of the bullet list the best sentence belongs to, headed by the list's heading."""
        fallback = best.chunk.heading_path.split(" > ")[-1]
        sibs = [(s, c or fallback) for s, g, c in split_with_context(best.chunk.text) if g == best.group]
        requirements = bool(_REQUIREMENTS_ASK.search(q.raw))
        items = [s for s, _ in sibs]
        # a requirements list is answered whole; place lists inside it ('Illinois: A, B, C, D') are not requirements
        if requirements:
            short = [x for x in items if not (x.count(",") >= 4 and len(x) > 60)]
            items = short if len(short) >= 3 else items
        sents = items[:_REQUIREMENTS_CAP if requirements else _LIST_CAP]
        if sibs and sibs[0][1] and sibs[0][1].lower() not in " ".join(sents).lower():
            sents.insert(0, sibs[0][1] + ":")
        if _AREA_WORDS.search(" ".join(sents)):
            # an item points at a place it does not name: add the page's own list of places
            for x, _ in split_sentences(best.page.parsed.text or ""):
                if x not in sents and _PLACE_LIST.search(x) and len(x) < 160:
                    sents.append(x.strip(" -*#"))
                    break
        return sents

    @staticmethod
    def _definition(best: Cand) -> list[str]:
        """A definition continues in the next sentences."""
        sents = [best.sent]
        flat = [x for x, _ in split_sentences(best.chunk.text)]
        if best.sent in flat:
            at = flat.index(best.sent)
            for nxt in flat[at + 1:at + 3]:
                if sum(len(x) for x in sents) + len(nxt) <= 420:
                    sents.append(nxt)
        return sents

    @staticmethod
    def _page_companions(best: Cand, cands: list[Cand]) -> list[str]:
        """The best sentence plus the best other sentences of its page, in page order."""
        sents = [best.sent]
        same = [c for c in cands if c.page.url == best.page.url and c.sent != best.sent and not c.teaser
                and c.score >= (_SAME_CHUNK_SHARE if c.chunk is best.chunk else _SAME_PAGE_SHARE) * best.score]
        same.sort(key=lambda c: -c.score)
        for c in same[:2]:
            if sum(len(x) for x in sents) + len(c.sent) <= 360:
                sents.append(c.sent)
        order = {s: i for i, (s, _) in enumerate(split_sentences(best.chunk.text))}
        if all(x in order for x in sents):
            sents.sort(key=lambda x: order[x])
        return sents

    @staticmethod
    def _finish(q, best: Cand, sents: list[str]) -> str:
        """Joins and caps the sentences; an article's title prefixes a sentence that does not say what it is about."""
        out = " ".join(sents)
        cap = _LONG_LIST_CAP if best.group >= 0 and len(sents) > _LIST_CAP else _DEFAULT_CAP
        if len(out) > cap:
            out = out[:cap - 3].rsplit(" ", 1)[0] + "..."
        title = strip_title_suffix(best.page.title or "").strip()
        asked = set(question_terms(q.raw))
        if (
            title
            and not title.startswith("http")
            # the sentence covers the question enough to stand without its release title
            and best.cov < 0.8
            and len(set(tokenize(out)) & asked) < len(asked)
            and best.page.page_type in ("newsroom", "blog")
            and title.lower() not in out.lower()
        ):
            out = f"{title}: {out}"
        return out

    # ------------------------------------------------------------------ several candidate answers
    def variants(self, q, cands: list[Cand]) -> Result | None:
        """'Do all branches keep the same hours?': show each schedule on file."""
        if not (_SAME_OR_DIFFERENT.search(q.raw) and _HOURS_WORD.search(q.raw)):
            return None
        seen, parts, srcs = set(), [], []
        for c in cands[:14]:
            if not HOURS_RX.search(c.sent):
                continue
            text = self.compose(q, c, cands)
            key = re.sub(r"[^0-9apm:]", "", text.lower().replace("noon", "12pm"))
            if key in seen:
                continue
            seen.add(key)
            parts.append(text)
            srcs.append(SourceRef(c.page.raw_url, c.page.snapshot_id, c.sent[:200]))
            # distinct schedules shown for 'do all branches keep the same hours?'
            if len(parts) == 4:
                break
        if len(parts) < 2:
            return None
        return Result("Hours differ by location. " + " | ".join(parts), srcs, True, None)

    def newest_near_duplicate(self, best: Cand, cands: list[Cand], q) -> Cand:
        """Among sentences that say nearly the same thing on different pages, the newest publish date wins."""
        group = [
            c
            for c in cands
            if c is best
            or (
                c.page.url != best.page.url
                and similarity(c.sent, best.sent) >= _NEAR_DUPLICATE_SIM
                and c.score >= _NEAR_DUPLICATE_SHARE * best.score
            )
        ]
        dated = [c for c in group if c.pub]
        if q.year_hint and len(group) > 1:
            # a year was asked for: the year-aware score already chose among recurring releases
            return best
        if len(group) > 1 and dated:
            return max(dated, key=lambda c: (c.pub, c.score))
        # the same sentence on several pages: the most apt page
        twins = [c for c in cands if c.sent == best.sent]
        return max(twins, key=lambda c: self._aptness(c, q)) if len(twins) > 1 else best

    def _body_tokens(self, page) -> set:
        if page.url not in self._page_tokens:
            self._page_tokens[page.url] = set(tokenize(page.parsed.text or ""))
        return self._page_tokens[page.url]

    def _aptness(self, c: Cand, q) -> tuple:
        asked = set(question_terms(q.raw))
        url_words = path_only(c.page.url).replace("/", " ").replace("-", " ")
        slug = [t for t in tokenize(url_words) if len(t) >= 4]
        body = self._body_tokens(c.page)
        self_fit = sum(1 for t in slug if t in body) / (len(slug) or 1)
        asked_year = int(bool(q.year_hint) and self.years.of(c.page) == q.year_hint)
        return (
            asked_year,
            len(asked & (set(tokenize(c.page.title or "")) | set(tokenize(url_words)))),
            round(self_fit, 2),
            c.score,
        )

