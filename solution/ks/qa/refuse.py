"""Where the system declines to answer, and why."""
from __future__ import annotations

import re

from ..core.tokens import tokenize as base_tokenize
from ..site_profile import INSTITUTION, links_off_site
from .answer_type import TypeBoost, shows_expected_detail
from .vocab import ANSWER_TYPE_WORDS, CANON, OK_UNKNOWN, STOP_QUESTION_WORDS, tokenize


# refusals the reader may still rescue: the extractive scorer found nothing good, not "the site does not say"
RESCUABLE = re.compile(
    "|".join(
        re.escape(x)
        for x in (
            "no sentence covers",
            "best sentence covers only",
            "lacks the kind of detail",
            "covers too little",
            "no figure in the retrieved",
            "no search hits",
            "only general-interest",
        )
    )
)

# ---------------------------------------------------------------------------------------------------- scope
_MARKETS = (r"stocks?|bonds?|etfs?|crypto\w*|bitcoin|ethereum|funds?|shares|commodit\w+|forex|options|futures|gold"
            r"|penny stocks?")
_ADVICE = re.compile(
    rf"\b(?:which|what)\b.{{0,25}}\b(?:{_MARKETS})\b.{{0,30}}\b(?:buy|sell|invest|pick|choose|trade)\b"
    rf"|\bshould i\b.{{0,40}}\b(?:buy|sell|invest|trade|short|bet|put|move|park)\b.{{0,40}}\b(?:{_MARKETS}|index fund"
    rf"s?|mutual funds?|the stock market)\b"
    rf"|\bshould i\b.{{0,30}}\b(?:buy|sell|invest in|trade|short|bet)\b"
    rf"|\b(?:is it|would it be|is now)\b.{{0,15}}\b(?:smart|wise|good|a good (?:idea|time))\b.{{0,30}}\b(?:invest|buy"
    rf"|sell|trade)\b.{{0,30}}\b(?:{_MARKETS})\b"
    rf"|\bbest (?:day|time|month) to (?:buy|sell|invest)\b"
    rf"|\b(?:stock|investment|market) (?:tips?|advice|predictions?|picks?)\b"
    r"|\bwill\b.{0,25}\b(?:rates?|market|prices?|economy|inflation)\b.{0,15}\b(?:go|rise|fall|drop|increase|decrease"
    r"|crash|change|be higher|be lower)\b",
    re.I,
)
_FORECAST = re.compile(
    r"(?=.*\b(?:rates?|apy|apr|dividends?|prices?|fees?|payments?)\b)\b(?:will|going to|gonna|predict\w*|forecast\w*"
    r"|project\w*|expect\w*)\b.{0,70}"
    r"\b(?:next (?:month|week|year|quarter)|in the future|tomorrow|soon|by (?:the )?end of (?:the )?(?:year|month"
    r"|quarter)|in \d+ (?:months?|years?))\b",
    re.I,
)
# a product name that is also an everyday word names the product only when one of its context words is present: (word,
# other-sense context, product context)
_AMBIGUOUS_WORDS = [
    (
        re.compile(r"\bvisa\b", re.I),
        re.compile(
            r"\b(?:student|tourist|work|immigration|passport|embassy|consulate|citizenship|permit|green card|entry"
            r"|travel|renew\w*|overseas|abroad)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:card|cards|credit|debit|platinum|rewards?|cash ?back|balance transfer|apr|rate|fee|limit"
            r"|mastercard)\b",
            re.I,
        ),
    ),
    (
        re.compile(r"\bpermit\b", re.I),
        re.compile(r"\b(?:work|residence|residency|immigration|travel|driving|fishing|hunting|parking)\b", re.I),
        re.compile(r"\b(?:business|building|construction|licen[sc]e|zoning)\b", re.I),
    ),
    (
        re.compile(r"\bcertificates?\b", re.I),
        re.compile(
            r"\b(?:birth|death|marriage|stock|gift|medical|diploma|authenticity|naturali[sz]ation|citizenship|baptism"
            r"|tax)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:deposit|cds?|apy|dividend|term|month|months|year|special|jumbo|ira|early withdrawal|penalty)\b", re.I
        ),
    ),
]


def out_of_scope(text: str) -> str | None:
    """Why a question is out of scope (advice, forecast, or an everyday word), else None."""
    if _ADVICE.search(text) or _FORECAST.search(text):
        return "asks for investment advice or a market prediction"
    for word, other, product in _AMBIGUOUS_WORDS:
        if word.search(text) and other.search(text) and not product.search(text):
            return "uses a product name in another sense"
    return None


# ---------------------------------------------------------------------------------------------------- data
def beyond_the_data(clean: str, latest: str, kind_of, res) -> str | None:
    if not res.confident or not latest[:4].isdigit():
        return None
    years = [y for y in re.findall(r"\b(20\d\d)\b", clean) if int(y) > int(latest[:4])]
    if not years or kind_of() in ("rate", "fee", "min_open", "min_earn", "contact"):
        return None
    evidence = res.text + " " + " ".join(s.quote or "" for s in res.sources)
    if all(y in evidence for y in years):
        return None
    return f"the site has nothing dated {years[0]} yet"


_NUMBER_NAMES = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    15: "fifteen",
    20: "twenty",
    30: "thirty",
}


def _mentions_term(low: str, months: int) -> bool:
    """Does the text name this term, as months or as years ('84 month', '7-year', 'seven year')?"""
    forms = [(months, "month|mo")]
    if months % 12 == 0:
        forms.append((months // 12, "year|yr"))
    for n, unit in forms:
        pat = rf"\b(?:{n}{('|' + _NUMBER_NAMES[n]) if n in _NUMBER_NAMES else ''})[\s-]*(?:{unit})s?\b"
        if re.search(pat, low):
            return True
    return False


def unpublished_variant(q, res, missing) -> str | None:
    if not (res.confident and q.kind in ("rate", "min_open", "min_earn") and q.intent != "url"):
        return None
    low = res.text.lower()
    term = q.qualifiers.get("term_months")
    if any(w.lower() not in low for w in missing) or (isinstance(term, int) and not _mentions_term(low, term)):
        return "the question names a variant (term or type) that no published rate carries"
    return None


# ---------------------------------------------------------------------------------------------------- text path
def unknown_terms(raw: str, idf) -> list[str]:
    """Content words of the question that occur nowhere on the site (a topic it does not publish)."""
    out = []
    for w in re.findall(r"[A-Za-z]{5,}", raw):
        lw = w.lower()
        if lw in OK_UNKNOWN:
            continue
        toks = base_tokenize(lw)
        if toks and all(t not in idf and CANON.get(t, t) not in idf for t in toks):
            out.append(lw)
    return out


def unknown_entity(raw: str, idf) -> str | None:
    """A capitalised word (or one beside 'branch'/'location') that occurs in no page of the index."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]+", raw)
    low = [w.lower() for w in words]
    cand = []
    for i, w in enumerate(words):
        capital = w[0].isupper() and i > 0 and not w.isupper()
        near_place = (i + 1 < len(low) and low[i + 1] in ("branch", "location", "office", "atm", "avenue", "street")) \
            or (i > 0 and low[i - 1] in ("at", "in") and i + 1 < len(low) and low[i + 1] in ("branch", "location"))
        if capital or near_place:
            cand.append(w.lower())
    for lw in cand:
        toks = tokenize(lw)
        if not toks or toks[0] in STOP_QUESTION_WORDS or len(lw) < 4:
            continue
        if all(t not in idf for t in toks):
            return lw
    return None


def entity_gap(q, ents: set, best, hits) -> str | None:
    seen = (
        set(tokenize(best.sent))
        | set(tokenize(best.chunk.text))
        | set(tokenize(best.page.title or ""))
        | set(tokenize(best.chunk.heading_path))
    )
    if q.intent == "branch":
        near = set(tokenize(best.sent)) | set(tokenize(best.page.title or "")) | set(tokenize(best.chunk.heading_path))
        if not (near & {"branch", "office", "location", "atm", "lobby"}) or (ents - near):
            return f"no branch passage names {sorted(ents - near) or sorted(ents)}"
    else:
        for h in hits[:8]:
            for ch, _ in h.chunks:
                seen |= set(tokenize(ch.text))
    absent = ents - seen
    return f"the retrieved text never mentions {sorted(absent)}" if absent else None


_NAMES_INSTITUTION = re.compile(rf"\b{re.escape(INSTITUTION)}\b", re.I)


def link_only(text: str) -> bool:
    """The answer is little more than a link to another organisation."""
    bare = re.sub(r"\([^)]*\)|https?://\S+", " ", text)
    return links_off_site(text) and len(re.findall(r"[A-Za-z]{3,}", bare)) < 8


def general_interest_only(q, best, text: str) -> bool:
    if _NAMES_INSTITUTION.search(text):
        return False
    page_type = best.page.page_type
    if page_type in ("blog", "newsroom") and re.match(r"\s*(?:does|do|can|could|is|are|will)\b", q.raw, re.I):
        return True
    return page_type == "blog" and not re.match(
        r"\s*(?:what(?:'s| is| are)|who|why|explain|define|describe)\b", q.raw, re.I
    )


def supported_by_chunk(q, best, terms) -> bool:
    if not terms or TypeBoost(q).score(best.sent) <= 0:
        return False
    have = set(tokenize(best.chunk.text)) | set(tokenize(best.page.title or ""))
    return sum(1 for t in terms if t in have) / len(terms) >= 0.5


# ---------------------------------------------------------------------------------------------------- sufficiency of
# the answer text
_MARKET_ACTION = re.compile(
    r"\b(?:can|could|may|do|does|will|am i able to|is it possible to)\b.{0,25}?\b(trade|buy|sell|invest|speculate"
    r"|mine|short)\b",
    re.I,
)
_BOUGHT_THING = re.compile(r"\b(?:car|cars|vehicle|truck|boat|rv|motorcycle|home|house|property|land|camper)\b", re.I)
_LIMIT_ASK = re.compile(r"\b(?:limits?|maxi?mum|max|cap)\b", re.I)
_LIMIT_WORDS = {"limit", "limits", "maximum", "cap"}
_GENERIC_SUBJECT_WORDS = _LIMIT_WORDS | {
    "much",
    "what",
    "amount",
    "card",
    "cards",
    "account",
    "accounts",
    "loan",
    "loans",
    INSTITUTION.lower(),
    "daily",
    "monthly",
}
_PERSON_FACT = re.compile(r"\b(?:born|birth ?(?:day|date|year|place)|died|net worth|salary)\b", re.I)
_ROLE = re.compile(r"\b(founders?|ceo|president|chair\w*|manager|director|employees?|executive|leader|owner)\b")


def _asks_unmentioned_market_action(q, mentioned: set) -> bool:
    """'Can I trade crypto?': the action asked (not buying a car or home, which is what loans are for)"""
    verb = _MARKET_ACTION.search(q.raw)
    if verb and _BOUGHT_THING.search(q.raw):
        verb = None
    return bool(verb) and verb.group(1).lower()[:4] not in {x[:4] for x in mentioned}


def _limit_subject_missing(q, terms, best, idf) -> bool:
    if not _LIMIT_ASK.search(q.raw):
        return False
    subject = [t for t in terms if idf.get(t, 0) >= 3.0 and t.isalpha() and len(t) >= 4 and t not in _LIMIT_WORDS]
    sent5 = {x[:5] for x in tokenize(best.sent)}
    raw_terms = [
        x for x in tokenize(q.raw) if x.isalpha() and len(x) >= 4 and x in idf and x not in _GENERIC_SUBJECT_WORDS
    ]
    chunk5 = {x[:5] for x in tokenize(best.chunk.text)}
    if raw_terms and max(raw_terms, key=lambda x: idf[x])[:5] not in chunk5:
        return True
    return bool(subject) and not {x[:5] for x in subject} & sent5


def _person_role_missing(q, best) -> bool:
    """A fact about a person (born, died, salary): the sentence must be about that role."""
    if not _PERSON_FACT.search(q.raw):
        return False
    role = _ROLE.findall(q.raw.lower())
    return bool(role) and not any(r[:5] in best.sent.lower() for r in role)


def is_sufficient(q, text: str, best, terms, idf, strict: bool = False) -> bool:
    """Does the answer text (with its page title and heading) cover enough of the question's content words?"""
    if len(terms) <= 2:
        return True
    have = set(tokenize(text)) | set(tokenize(best.page.title or "")) | set(tokenize(best.chunk.heading_path))
    # a year-specific statistic of a dated release: its lead names the program, the sentence only the figure
    if q.year_hint and best.page.page_type in ("newsroom", "blog"):
        have |= set(tokenize(best.chunk.text))
    top = max(idf.values()) if idf else 1.0
    # satisfied by the answer type, checked separately
    terms = [t for t in terms if t not in ANSWER_TYPE_WORDS] or terms
    w = {t: idf.get(t, top) for t in terms}
    stems = {x[:6] for x in have if len(x) >= 6}
    chunk_toks = set(tokenize(best.chunk.text))
    if (
        _asks_unmentioned_market_action(q, chunk_toks | have)
        or _limit_subject_missing(q, terms, best, idf)
        or _person_role_missing(q, best)
    ):
        return False
    gone = [t for t in terms if t not in have and t not in chunk_toks and not (len(t) >= 6 and t[:6] in stems)]
    distinctive = [t for t in gone if t.isalpha() and len(t) >= 5 and idf.get(t, 0) >= 4.4 and t not in OK_UNKNOWN]
    # the passage has the kind of detail asked for
    typed = shows_expected_detail(q, text)
    total = sum(w.values()) or 1.0
    # the question is mostly about something else
    if distinctive and sum(w[t] for t in gone) / total >= (0.45, 0.7)[typed]:
        return False
    share = sum(w[t] for t in terms if t in have or (len(t) >= 6 and t[:6] in stems)) / total
    if share < (0.4, 0.3)[typed]:
        return False
    rare = max((t for t in terms if t.isalpha() and len(t) >= 4), key=lambda t: w[t], default=None)
    # the question's rarest content word must be in the passage
    if strict and rare is not None and idf.get(rare, 0) > 0:
        return rare in have or rare in chunk_toks
    return True
