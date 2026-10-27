"""Text answers of a special shape: product lists of a topic, site figures and counts, and pointers to a page or service."""
from __future__ import annotations

import re

from ..core.textutil import MD_LINK_URL, PHONE_DASHED
from ..core.types import Result, SourceRef
from ..ingest.site_stats import best_box, box_wins
from ..site_profile import (
    INSTITUTION,
    NATIONAL_COUNT_MARKERS,
    NATIONAL_FIGURE_MARKERS,
    SITE_FIGURES,
    is_own_site,
)
from .sentences import split_sentences
from .vocab import (
    HEADING_SKIP,
    LINK_SYNONYMS,
    NOUN_GROUPS,
    PRODUCT_WORD,
    STAT_WORDS,
    STOP_QUESTION_WORDS,
    question_terms,
    tokenize,
)

# ---- 'What X does it offer?' answers: product headings of a topic's hub page and sibling product pages.
LIST_ASK = re.compile(
    r"\b(?:what|which)\s+(?:(?:kinds?|types?|sorts?|forms?|categories)\s+of\b|(?:[\w-]+\s+){0,2}?[\w-]+s\s+(?:does|do"
    r"|can|are|is|will)\b.*\b(?:offer|support|accept|have|available|provide|work with|use|take)\b)|\blist (?:all "
    r"|the |of )?",
    re.I,
)
_ENUM_ITEM = r"[A-Z][\w'’&.-]*(?: [A-Z][\w'’&.-]*)*"
_ENUM = re.compile(rf"{_ENUM_ITEM}(?:,\s*{_ENUM_ITEM})+,?\s+(?:and|or)\s+{_ENUM_ITEM}")
_PERSON = re.compile(r"^[A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z’'-]+)+$")
_NOT_A_PRODUCT = re.compile(r"\b(?:protection|insurance|calculators?)\b", re.I)
_NOT_A_PRODUCT_LENIENT = re.compile(r"\b(?:rates|faq|questions?|calculator)\b", re.I)
_ENUMERATION_ANCHORS = re.compile(rf"\b{re.escape(INSTITUTION)}\b|\byour\b|\bour\b")
_ENUMERATION_STARTS = ("The", "A", "An", INSTITUTION)
_NOT_TOPIC = {
    "kind",
    "type",
    "sort",
    "form",
    "categorie",
    "offer",
    "support",
    "accept",
    "available",
    "provide",
    "list",
    "different",
    "all",
    "use",
    "take",
    "work",
}
# 4-letter stems of words every hub URL or heading has
_GENERIC_TOPIC_STEMS = {"busi", "pers", "acco", "prod", "opti", "avai"}
_NOT_A_HUB_SECTION = re.compile(r"protection|insurance|calculator|rates?|apply|application|resources?")
_ENUMERATION_HITS, _ENUMERATION_CHUNKS = 4, 4


class HubPages:
    """The product hub pages of a snapshot view: one page per product family, with a heading per product."""

    def __init__(self, kb):
        self.kb = kb

    def of_question(self, q, topic: set, business: bool):
        """(hub page, its product names) for a family on the question's snapshot; (None, []) when none."""
        view = self.kb.view_for(q.as_of)
        if view is None:
            return None, []
        return self.find(view, topic, business)

    @staticmethod
    def _headings(lines: list[str], prefix: str, lenient: bool = False) -> list[str]:
        """Product-like headings of a level, in page order, without repeats."""
        found = [re.sub(r"^#+\s*", "", ln).strip(" *") for ln in lines if re.match(prefix, ln)]
        return [
            h
            for h in dict.fromkeys(found)
            if 2 < len(h) <= 40
            and not h.endswith("?")
            and (
                not HEADING_SKIP.search(h)
                or (lenient and PRODUCT_WORD.search(h) and not _NOT_A_PRODUCT_LENIENT.search(h))
            )
            and not (_PERSON.match(h) and not PRODUCT_WORD.search(h))
            and not _NOT_A_PRODUCT.search(h)
        ]

    def _products_of(self, page) -> list[str] | None:
        lines = (page.parsed.text or "").split("\n")
        h3 = self._headings(lines, r"###\s", True)
        named = [h for h in h3 if PRODUCT_WORD.search(h)]
        # 'Digital Banking' / 'Everyday Essentials' are not products
        heads = named if len(named) >= 2 else h3
        if len(heads) < 2:
            heads = self._headings(lines, r"#{2,4}\s")
        if len(heads) < 3 and len(h3) < 2:
            return None
        return heads

    def find(self, view, topic, want_business):
        """The hub page whose URL names the topic most tightly, and its product names."""
        best, best_key, items = None, None, []
        t4 = {x[:4] for x in topic if x}
        core4 = t4 - _GENERIC_TOPIC_STEMS or t4
        for page in sorted(view.pages.values(), key=lambda p: p.url):
            if page.page_type not in ("product", "rate_sheet") or page.parsed.listing:
                continue
            path = re.sub(r"https?://[^/]+", "", page.url).strip("/")
            slug = tokenize(path.replace("/", " ").replace("-", " "))
            slug4 = {x[:4] for x in slug}
            if not t4 or not (slug4 & core4) or ("business" in slug) != want_business or path.count("/") > 1:
                continue
            heads = self._products_of(page)
            if heads is None:
                continue
            key = (
                len(set(slug) - {"personal", "business"} - set(topic)) == 0 or len(slug4 - {"pers", "busi"} - t4) <= 1,
                len(heads),
            )
            if best_key is None or key > best_key:
                best, best_key, items = page, key, heads
        return best, items


def _enumeration(hits) -> Result | None:
    found = []
    for rank, h in enumerate(hits[:_ENUMERATION_HITS]):
        for chunk, _cs in h.chunks[:_ENUMERATION_CHUNKS]:
            for sent, _g in split_sentences(chunk.text):
                flat = re.sub(r"\(https?://[^)]*\)", "", sent)
                if not _ENUMERATION_ANCHORS.search(flat):
                    continue
                for m in _ENUM.finditer(flat):
                    names = [x.strip() for x in re.split(r",\s*|\s+(?:and|or)\s+", m.group(0)) if x.strip()]
                    # fewer names than this is not a list
                    if len(names) >= 3 and not any(x.split()[0] in _ENUMERATION_STARTS for x in names):
                        found.append((rank, -len(names), h.page.url, sent.strip(), h.page))
    if not found:
        return None
    found.sort(key=lambda t: t[:4])
    sent, page = found[0][3], found[0][4]
    return Result(sent, [SourceRef(page.raw_url, page.snapshot_id, sent[:200])], True, None)


def _sibling_products(view, topic: set, want_business: bool, items: list[str]) -> None:
    """Adds, to the hub's product names, the sibling product pages of the topic that the hub does not mention."""
    core_topic = topic - {"business", "personal"}
    for page in sorted(view.pages.values(), key=lambda p: p.url):
        path = re.sub(r"https?://[^/]+/", "", page.url)
        if (
            page.page_type not in ("product", "rate_sheet")
            or path.count("/") < 2
            or not path.startswith("business/" if want_business else "personal/")
        ):
            continue
        slug = tokenize(path.split("/")[-1].replace("-", " "))
        whole = tokenize(path.replace("/", " ").replace("-", " "))
        if not ({x[:4] for x in whole} & {x[:4] for x in core_topic}) or _NOT_A_HUB_SECTION.search(path):
            # under the topic's directory or named for it
            continue
        covered = {x[:4] for i in items for x in tokenize(i)}
        # the hub already names it ('Equipment & Vehicle Loan')
        if {x[:4] for x in slug} - {x[:4] for x in topic} - {"busi", "pers"} <= covered:
            continue
        label = " ".join(w.capitalize() for w in path.split("/")[-1].split("-"))
        if label not in items:
            items.append(label)


def list_answer(kb, hubs: HubPages, q, hits) -> Result | None:
    view = kb.view_for(q.as_of)
    topic = {t for t in question_terms(q.raw) if t.isalpha()} - _NOT_TOPIC
    listed = _enumeration(hits)
    if listed is not None:
        return listed
    want_business = "business" in topic
    page, items = hubs.find(view, topic, want_business)
    if page is None:
        return None
    _sibling_products(view, topic, want_business, items)
    text = f"{kb.voice.poss()} {' '.join(sorted(topic))} options: " + "; ".join(items) + "."
    return Result(text, [SourceRef(page.raw_url, page.snapshot_id, "; ".join(items)[:200])], True, None)

# ---- Site-level figures (members, employees, branches...) and counts the pages state.
# a longer "sentence" is a run of page text, not a statement
_MAX_SENTENCE = 300
# characters kept after a figure when its sentence has no end
_OPEN_TAIL = 80
_STAT_ASK = re.compile(
    r"\b(?:how (?:many|much|big|large)|number of|total|count|size|what (?:is|are)|when|(?:what|which) year)\b", re.I
)
_UNSTATED_TOPICS = re.compile(
    r"\b(?:total deposits|deposits|total loans outstanding|total outstanding loans|loans outstanding|outstanding "
    r"loans|total loans|loan portfolio)\b",
    re.I,
)
_TEAM_MEMBERS = re.compile(r"\bteam members?\b", re.I)
_SENTENCE_END = re.compile(r"\.(?=\s|$)")
_FIGURES = tuple((f.name, re.compile(f.asked, re.I), re.compile(f.stated, re.I)) for f in SITE_FIGURES)
_NATIONAL = re.compile(NATIONAL_FIGURE_MARKERS)
_NATIONAL_COUNT = re.compile(NATIONAL_COUNT_MARKERS)
_COUNTED_ON = ("newsroom", "other", "product", "branch_contact")
_COUNT_VERBS = r"(?:does|do|did|has|have|are|is|were|was|runs?|operates?|maintains?|owns?|offers?|serves?|there)"
_HOW_MANY = re.compile(rf"\bhow many ([\w-]+(?: [\w-]+){{0,2}}?) {_COUNT_VERBS}\b", re.I)
_NUMBER_OF = re.compile(
    rf"\b(?:the )?(?:total )?(?:number|count) of ([\w-]+(?: [\w-]+){{0,2}}?)(?= {_COUNT_VERBS}\b| (?"
    rf":{re.escape(INSTITUTION.lower())}|in|at|across)\b|\s*[?.!]?\s*$)",
    re.I,
)
_COUNT_VERB_WORDS = {"number", "count", "total", "run", "operate", "maintain", "own", "serve", "have", "offer"}


def _statement_around(text: str, m: re.Match) -> str:
    """The sentence (or line) of a boilerplate statement."""
    start = max(text.rfind(".", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
    end = _SENTENCE_END.search(text, m.end())
    stop = end.start() if end else -1
    return " ".join(text[start:stop if stop > 0 else m.end() + _OPEN_TAIL].split()).lstrip(" -")


def _asked_figure(q) -> tuple[str, re.Pattern] | None:
    """(topic, statement pattern) when the question asks for exactly one site-level figure."""
    if q.year_hint or q.month_hint or q.intent in ("value", "check", "change"):
        return None
    raw = q.raw
    if not _STAT_ASK.search(raw):
        return None
    plain = _UNSTATED_TOPICS.sub(" ", raw)
    left = [
        w
        for w in re.findall(r"[a-z]+", plain.lower())
        if w not in STAT_WORDS and w not in STOP_QUESTION_WORDS and len(w) > 1
    ]
    if left:
        # the question asks about something more specific than the site-level figure
        return None
    asked = _TEAM_MEMBERS.sub("staff", raw)
    topics = [(name, stated) for name, asked_rx, stated in _FIGURES if asked_rx.search(asked)]
    # 'membership' is also a modifier ('field of membership', 'members at each branch')
    if len(topics) > 1:
        topics = [t for t in topics if t[0] != "members"]
    return topics[0] if len(topics) == 1 else None


def _newest_statement(view, stated: re.Pattern):
    """((publish date, official, url), sentence, page) of the newest page that states the figure, or None."""
    best = None
    for page in view.pages.values():
        # page types that never carry the boilerplate statement
        if page.page_type in ("rates", "branch_contact") or not page.publish_date:
            continue
        text = page.parsed.text or ""
        for m in stated.finditer(text):
            sent = _statement_around(text, m)
            if len(sent) > _MAX_SENTENCE or _NATIONAL.search(sent):
                continue
            key = (page.publish_date, page.page_type != "newsroom", page.url)
            if best is None or key > best[0]:
                best = (key, sent, page)
    return best


def site_figure(kb, q) -> Result | None:
    """The site-level figure the question asks for, or None when it asks for anything else."""
    asked = _asked_figure(q)
    if asked is None:
        return None
    name, stated = asked
    view = kb.view_for(q.as_of)
    best = _newest_statement(view, stated)
    box = best_box(view, name)
    # the dated official box outranks boilerplate
    if box and box_wins(box[0], best[0][0] if best else None):
        b, label, value = box
        text = f"{label}: {value} (as of {b.date_text})"
        return Result(text, [SourceRef(b.raw_url, b.snapshot_id, f"{b.line} | {label}: {value}")], True, None)
    if best is None:
        return None
    _, sent, page = best
    return Result(sent, [SourceRef(page.raw_url, page.snapshot_id, sent[:200])], True, None)


def _count_sentence(text: str, m: re.Match) -> str:
    start = text.rfind(".", 0, m.start()) + 1
    end = text.find(".", m.end())
    return " ".join(text[start:end if end > 0 else m.end() + _OPEN_TAIL].split())


def count_answer(kb, q) -> Result | None:
    if q.year_hint:
        # a count for a given year is read from that year's release by the scored path
        return None
    m = _HOW_MANY.search(q.raw) or _NUMBER_OF.search(q.raw)
    if not m:
        return None
    noun = tokenize(m.group(1).split()[-1])
    if not noun:
        return None
    group = next((g for g in NOUN_GROUPS if noun[0] in g or noun[0].rstrip("s") in g), {noun[0]})
    stems = "|".join(sorted({re.escape(x[:5]) for x in group}))
    counted = re.compile(rf"(?<![$\d,.])\b(\d[\d,]*)\s+(?:[\w-]+\s+){{0,2}}(?:{stems})\w*", re.I)
    phrase = set(tokenize(m.group(1)))
    others = [
        t
        for t in question_terms(q.raw)
        if t not in group and t not in phrase and t.isalpha() and t not in _COUNT_VERB_WORDS
    ]
    found = []
    for page in kb.view_for(q.as_of).pages.values():
        if page.page_type not in _COUNTED_ON or not page.publish_date:
            continue
        text = page.parsed.text or ""
        for sm in counted.finditer(text):
            sent = _count_sentence(text, sm)
            if _NATIONAL_COUNT.search(sent) or len(sent) > _MAX_SENTENCE:
                continue
            toks = set(tokenize(sent))
            # share of the question's other words a counting sentence must carry
            if others and sum(1 for t in others if t in toks) / len(others) < 0.6:
                continue
            found.append((page.publish_date, page.url, sent, page))
    if not found:
        return None
    found.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, _, sent, page = found[0]
    return Result(sent.lstrip(" -"), [SourceRef(page.raw_url, page.snapshot_id, sent[:200])], True, None)

# ---- Answers that point the member somewhere: a page, a phone number, a service request
_SERVICE_REQUEST = re.compile(
    r"\s*(?:how|where|what)\b.*\b(?:update|change|correct|edit|modify)\b.*\b(?:contact|address|phone|email|e-mail)\b",
    re.I,
)
# a link word in the same synonym group as a question word counts for half a match
_SYNONYM_CREDIT = 0.5
# share of a link's score taken from how well its passage ranked
_RANK_WEIGHT = 0.1
_LINK_HITS, _LINK_CHUNKS, _PAGE_LINK_HITS = 8, 2, 5


def _words(url: str) -> str:
    return re.sub(r"https?://(?:www\.)?", "", url).replace("/", " ").replace(".", " ").replace("-", " ")


def link_answer(q, hits, terms: list[str]) -> Result | None:
    """The link whose anchor text and URL best match the question."""
    qset = set(terms)
    groups = [g for g in LINK_SYNONYMS if g & qset]
    top = hits[0].score or 1
    scored = []
    for h in hits[:_LINK_HITS]:
        for chunk, cs in h.chunks[:_LINK_CHUNKS]:
            for m in MD_LINK_URL.finditer(chunk.text):
                anchor, url = m.group(1).strip(), m.group(2)
                if not url.startswith("http") or any(
                    x in url for x in ("/hubfs/", ".jpg", ".png", ".svg", "javascript")
                ):
                    continue
                toks = set(tokenize(anchor + " " + _words(url)))
                overlap = len(toks & qset) + sum(_SYNONYM_CREDIT for g in groups if toks & g)
                if overlap:
                    scored.append(
                        (
                            # a link on the institution's own site beats one that leaves it
                            overlap + (0.6 if is_own_site(url) else 0.0) + _RANK_WEIGHT * (cs / top),
                            url,
                            anchor,
                            h,
                        )
                    )
    # the page itself is a candidate link
    for h in hits[:_PAGE_LINK_HITS]:
        toks = set(tokenize(_words(h.page.url) + " " + h.page.title))
        overlap = len(toks & qset) + sum(_SYNONYM_CREDIT for g in groups if toks & g)
        if overlap:
            scored.append(
                # a retrieved page is itself a candidate link
                (overlap + 0.6 + _RANK_WEIGHT * (h.score / top), h.page.raw_url, h.page.title, h)
            )
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    _, url, anchor, h = scored[0]
    text = (
        f"{anchor}: {url}"
        if anchor and not anchor.startswith("http") and len(anchor) < 80
        else f"You can find that here: {url}"
    )
    return Result(text, [SourceRef(h.page.raw_url, h.page.snapshot_id, anchor[:200])], True, None)


def service_request(kb, q) -> Result | None:
    if not _SERVICE_REQUEST.match(q.raw):
        return None
    for page in sorted(kb.view_for(q.as_of).pages.values(), key=lambda p: p.url):
        if page.url.rstrip("/").endswith("/contact"):
            m = PHONE_DASHED.search(page.parsed.text or "")
            if m:
                text = (
                    f"I did not find step-by-step instructions for that, but {kb.voice.poss()} contact page lists"
                    f" {m.group(0)} for member service."
                )
                return Result(
                    text,
                    [SourceRef(page.raw_url, page.snapshot_id, m.group(0))],
                    False,
                    "no procedure published; contact page phone given",
                )
    return None

