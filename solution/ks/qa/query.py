"""Question understanding: one parser turns a member's question into a structured `Query`: dates (absolute and
relative) are read and cut out, wording is normalised, the product qualifiers come from `ingest.normalize.parse_query`,
then kind, wanted attributes and intent are classified. The same module cleans the raw question first (filler removed, 'the X page
says' claims reduced, context split from the question)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..core.dates import MONTH_PATTERN, month_number, parse_as_of
from ..core.textutil import SENT_BREAK_CAPITAL_DIGIT
from ..ingest.normalize import parse_query
from ..site_profile import ACRONYM_EXPANSIONS, INSTITUTION, QUESTION_WORDING

# ---- dates in a question: absolute, relative ('last March') and ranges, as an as_of plus the remaining text
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "eighteen": 18,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
}
_RELATIVE_COUNT = {**NUMBER_WORDS, "a": 1, "an": 1, "couple": 2, "few": 3}

_DATE_SPANS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            rf"\b(?:as of|on|since|during|in|by|from|at)?\s*\(?\b(?:(early|mid|late|end of)[- ])?{MONTH_PATTERN}\.?\s"
            rf"+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\)?",
            re.I,
        ),
        "mdy",
    ),
    (re.compile(r"(?:\b(?:as of|on|since|from)\s+)?\b(20\d\d)-(\d\d)-(\d\d)\b", re.I), "iso"),
    (re.compile(r"(?:\b(?:as of|on|since|from)\s+)?\b(\d{1,2})/(\d{1,2})/(\d{4})\b", re.I), "num"),
    (
        re.compile(
            rf"\b(?:as of|in|during|since|from|by|for|at)?\s*\(?\b(?:(early|mid|late|end of)[- ])?{MONTH_PATTERN}\.?,"
            rf"?\s+(20\d\d)\)?",
            re.I,
        ),
        "my",
    ),
    (
        re.compile(
            rf"\b(?:as of|in|during|since|from|by|for|at)\s+(?:(early|mid|late|end of)[- ])?{MONTH_PATTERN}\b(?!\s*\d)",
            re.I,
        ),
        "m",
    ),
]
_RELATIVE_RULES = [
    (r"\bbefore\s+" + MONTH_PATTERN + r"\b(?:\s+(20\d\d))?", "before"),
    (r"\b(\d+|" + "|".join(_RELATIVE_COUNT) + r")\s+(day|week|month|year)s?\s+ago\b", "ago"),
    (r"\blast month\b", "last month"), (r"\blast week\b", "last week"), (r"\byesterday\b", "yesterday"),
]
# the day a 'mid-May' reads as; a bare month reads as its last day
_MONTH_POINT = {"early": 7, "mid": 15, "late": 25}
_NOW_WORDS = re.compile(r"\b(?:today|right now|currently|now)\s*(?:\(\s*\w*\.?\s*(?:20\d\d)?\s*\))?", re.I)


def _last_day(y: int, m: int) -> date:
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return nxt - timedelta(days=1)


def _shift_months(d: date, n: int) -> date:
    y, m = d.year, d.month - n
    while m <= 0:
        m, y = m + 12, y - 1
    return date(y, m, min(d.day, _last_day(y, m).day))


def _pick(y: int, mo: int, which: str | None) -> date:
    day = _MONTH_POINT.get((which or "").lower())
    return date(y, mo, day) if day else _last_day(y, mo)


def _relative_date(text: str, ref_iso: str):
    """('last month' / 'two weeks ago' / 'before June' ...) -> (as_of ISO, text without the phrase)."""
    try:
        ref = date.fromisoformat(ref_iso[:10])
    except ValueError:
        return None, text
    # a reference date beyond this is a placeholder, not a snapshot date
    if ref.year > 2100:
        return None, text
    for pat, kind in _RELATIVE_RULES:
        m = re.search(pat, text, re.I)
        if not m:
            continue
        if kind == "before":
            mo = month_number(m.group(1))
            y = int(m.group(2)) if m.group(2) else (ref.year if date(ref.year, mo, 1) <= ref else ref.year - 1)
            d = date(y, mo, 1) - timedelta(days=1)
        elif kind == "ago":
            n = int(m.group(1)) if m.group(1).isdigit() else _RELATIVE_COUNT[m.group(1).lower()]
            unit = m.group(2).lower()
            d = (
                ref - timedelta(days=n if unit == "day" else 7 * n)
                if unit in ("day", "week")
                else _shift_months(ref, n * (12 if unit == "year" else 1))
            )
        elif kind == "last month":
            d = _shift_months(ref, 1)
            d = _last_day(d.year, d.month)
        elif kind == "last week":
            d = ref - timedelta(days=7)
        else:
            d = ref - timedelta(days=1)
        return d.isoformat(), text[:m.start()] + " " + text[m.end():]
    return None, text


def _span_date(kind: str, g: tuple, latest: str, ly: int) -> date:
    """The date a matched span stands for (ValueError when it is not a real date)."""
    if kind == "mdy":
        return date(int(g[3]), month_number(g[1]), int(g[2]))
    if kind == "iso":
        return date(int(g[0]), int(g[1]), int(g[2]))
    if kind == "num":
        return date(int(g[2]), int(g[0]), int(g[1]))
    if kind == "my":
        return _pick(int(g[2]), month_number(g[1]), g[0])
    mo = month_number(g[1])
    y = ly if _last_day(ly, mo).isoformat() <= latest or date(ly, mo, 1).isoformat() <= latest else ly - 1
    return _pick(y, mo, g[0])


def _explicit_date(out: str, latest: str, ly: int):
    for rx, kind in _DATE_SPANS:
        m = rx.search(out)
        if not m:
            continue
        try:
            d = _span_date(kind, m.groups(), latest, ly)
        except ValueError:
            continue
        iso = d.isoformat()
        as_of = None if iso > latest and iso[:7] >= latest[:7] else iso
        if m.group(0).strip().lower().startswith("since"):
            # 'changed since May' asks about now, with history
            as_of = None
        return as_of, d.year, out[:m.start()] + " " + out[m.end():]
    return None, None, out


def _year_hint(out: str, hint: int | None, as_of: str | None, ly: int):
    """('in 2025' / a bare '2021' / 'last year' / 'this year') -> (hint, text without the phrase)."""
    vehicle = r"auto|car|vehicle|truck|suv|motorcycle|boat|rv"
    # '2024 model': a model year, not a date
    m = re.search(r"\b(?:in|for|during|of)\s+(20\d\d)\b(?!\s*(?:model|vehicle|car|truck|suv))", out, re.I)
    if m and as_of is None and not re.search(rf"\b({vehicle})\b", out, re.I):
        hint = int(m.group(1))
        out = out[:m.start()] + " " + out[m.end():]
    m = re.search(
        r"(?<![\$\d.,/-])\b(20[012]\d)\b(?![\d.,%/-]|\s*(?:model|vehicle|car|truck|suv|ford|toyota|honda|chevy"
        r"|chevrolet|bmw))",
        out,
        re.I,
    )
    if m and hint is None and as_of is None and not re.search(rf"\b({vehicle}|model)\b", out, re.I):
        # '... the 2021 bonus dividend': a bare year names the year the thing is about
        hint = int(m.group(1))
        out = out[:m.start()] + " " + out[m.end():]
    if re.search(r"\blast year\b", out, re.I):
        hint = ly - 1
        out = re.sub(r"\blast year\b", " ", out, flags=re.I)
    elif re.search(r"\bthis year\b", out, re.I):
        hint = ly
        out = re.sub(r"\bthis year\b", " ", out, flags=re.I)
    return hint, out


def date_from_text(text: str, latest: str, ref: str = "") -> tuple[str | None, int | None, str]:
    """(as_of ISO or None, year_hint, text without the date phrase)."""
    ly = int(latest[:4]) if latest else date.today().year
    rel, out = _relative_date(text, ref or latest)
    if rel:
        return rel, int(rel[:4]), out
    as_of, hint, out = _explicit_date(out, latest, ly)
    hint, out = _year_hint(out, hint, as_of, ly)
    if re.search(r"\((?:\w+\.? )?20\d\d\)", out):
        # 'today (May 2026)': the parenthesised date says it all
        out = _NOW_WORDS.sub(" ", out)
    return as_of, hint, out


# ---- the question
# kinds whose answer is a number from the fact store
NUMERIC_KINDS = ("rate", "fee", "min_open", "min_earn")

_RATE_FAMILIES = {
    "savings",
    "checking",
    "money_market",
    "certificate",
    "ira",
    "hsa",
    "credit_card",
    "mortgage",
    "auto_loan",
    "personal_loan",
    "heloc",
    "home_equity_loan",
    "home_improvement_loan",
    "recreational_loan",
}
_DEPOSIT_FAMILIES = {"savings", "checking", "money_market", "certificate", "ira", "hsa", ""}
_NUM_TERM = re.compile(rf"\b({'|'.join(NUMBER_WORDS)})[- ](year|month)s?\b", re.I)
_FEE_WORDS = re.compile(r"\b(fees?|charges?|costs?|cost|price|priced|how much (?:is|for|to|would it cost))\b", re.I)
_RATE_WORDS = re.compile(
    r"\b(rates?|apys?|aprs?|dividends?|interest|yield|earn(?:s|ing)?|(?<!apple )(?<!google )(?<!samsung )pays?|paying"
    r"|return|percent)\b|\d\s?%",
    re.I,
)
_RATE_FIGURE = re.compile(r"\b(?:rates?|apy|apr|dividends?|interest|yield|percent)\b|\d\s?%")
_OFFER_CONDITIONS = re.compile(r"\b(?:bonus|incentive|promo(?:tion)? code|giveback|gift)\b")
_BRAND = re.escape(INSTITUTION.lower())
_FEE_NOT_RATE = re.compile(
    rf"\b(fees?|price|priced|how much)\b|(?<!\bdoes )(?<!\bdo )(?<!\b{_BRAND} )(?<!\byou )\bcosts?\b"
)
_RATE_NOT_FEE = re.compile(r"\b(apr|apy|dividend|interest rate)\b")
_MIN_OPEN = re.compile(
    r"\bmin\w*\.?\s+(?:(?:opening |initial )?(?:deposit|balance|amount)\s+)?(?:to|for)\s+open\b|\bopening deposit\b|"
    r"\bminimum (?:opening |initial )?deposit\b|\bto open\b.*\bminimum\b|\bhow much (?:do i need )?to (?:open"
    r"|deposit)\b",
    re.I,
)
_MIN_EARN = re.compile(r"\bmin\w*\.?\s+(?:balance\s+)?to\s+earn\b", re.I)
_MEMBERSHIP = re.compile(r"\b(join|member|membership)\b")
_NEGATED_ATTR = r"\(?\b(?:not|excluding|without|rather than|instead of|other than)\s+(?:the\s+)?"
_NEGATED_RATE = re.compile(_NEGATED_ATTR + r"(apy|apr|dividend(?: rate)?|interest rate|rate)\b\)?", re.I)
_NEGATED_ATTR_NAMES = {"apy": "apy", "apr": "apr", "rate": "rate", "interest rate": "rate"}

_CHANGE = re.compile(
    r"\b(chang(?:e|ed|es|ing)|went (?:up|down)|go(?:es|ne)? (?:up|down)|increas\w+|decreas\w+|dropp?(?:ed)?|"
    r"fell|fallen|ris(?:e|en|ing)|rose|used to|previous(?:ly)?|history|historical|trend|compared (?:to|with)|"
    r"since|prior|earlier|was it|moved|update[ds]?|differ\w*)\b",
    re.I,
)
_NOT_A_CHANGE = re.compile(
    r"\b(change my|change the pin|change of address)\b|\bdifferent (?:\w+ ){0,3}(?:accounts?|cards?|loans?|products?"
    r"|types?|kinds?|options?|certificates?)\b"
)
_STILL = re.compile(r"\b(still|any ?more|no longer|currently|right now|as of today)\b", re.I)
_CHECK = re.compile(r"^\W*(?:do|does|did|is|are|was|were|can|could|will|would|has|have|am|should)\b", re.I)
_ANY = re.compile(r"\b(any|anything|either|some)\b", re.I)
_LIST = re.compile(
    rf"\b(all|every|each|list|options|choices|types? of|kinds? of|variet\w+|offerings?|compare|comparison|different|"
    rf"what (?:\w+ ){{0,4}}(?:can|could) (?:i|we) (?:get|choose|open|have|pick) from (?:{_BRAND}|you|the credit union)|"
    rf"which|available|breakdown|what (?:\w+ ){{0,3}}(?:do|does) (?:you|{_BRAND}|the credit union) (?:offer|have|prov"
    rf"ide|carry)|"
    rf"what (?:are|is) (?:the |your |{_BRAND}'?s )?(?:\w+ ){{0,2}}(?:terms|options|products|accounts|cards|rates))\b",
    re.I,
)
_PLURAL_PRODUCT = re.compile(r"\b(rates|accounts|cards|loans|certificates|terms|fees)\b")
_OFFER_WORDS = re.compile(r"\b(offer|have|available|options)\b")
_NOT_A_RATE_LIST = re.compile(r"\b(?:requirements?|eligib\w+|qualif\w+|conditions|bonus|catch|what do i need|steps)\b")
_NOT_A_RATE_ROW = re.compile(
    r"\b(how (?:do|can|to)|where|why|when|who|apply|eligib\w+|requirements?|link|url|bonus|catch|conditions|what do "
    r"i need|qualif\w+)\b"
)
_URLQ = re.compile(r"\b(link|url|web ?site|web ?page|webpage|page|site|portal|online)\b", re.I)
_LINK_NOUN = re.compile(r"\b(link|url|web ?site|web ?page|webpage)\b")
_LINK_OR_PAGE = re.compile(
    r"\b(link|url|web ?site|web ?page|webpage|address)\b|\bwhat page\b|\bwhich page\b|\bwhere (?:can|do|could|should)\b"
)
_LINK_VERB = re.compile(
    r"\b(?:can|could|do|to|would|will|should|may|cannot|can't)\s+(?:i|we|you|someone|members?|one)?\s*link\b"
    r"|\blink(?:ing|ed)?\s+(?:my|an?|the|with|your|another|external|other)\b",
    re.I,
)
_WHERE_TO_FIND = re.compile(
    r"\bwhere\b.*\b(see|find|view|look|check|posted|published|listed|located|get)\b|\bwhat page\b|\bwhich page\b"
)
_WHEREDO = re.compile(r"\bwhere (?:can|do|could|should|would) (?:i|we|you)\b|\bwhere (?:is|are) the\b", re.I)
_WHEREDO_ACTION = re.compile(r"\b(apply|join|sign ?up|log ?in|open|enroll|see|find|view|check|look)\b")
_LOCATOR = re.compile(r"\b(near me|nearest|closest|locator|find an? (?:branch|atm|location))\b")
_LINK_WORD = re.compile(r"\b(link|url)\b")
_HOWTO = re.compile(r"\b(how (?:do|can|could|would|should) (?:i|we|you)|how to|steps?|process|procedure|what do i need|"
                    r"requirements?|eligib\w+|qualif\w+|sign ?up|enroll|apply|report|what should i do)\b", re.I)
_BRANCH = re.compile(
    r"\b(branch(?:es)?|location|locations|atm|atms|lobby|drive[- ]?(?:up|through|thru)|hours?|open(?:s|ing)?\b|"
    r"clos(?:e|es|ed|ing|ure)|address|located|saturday|sunday|weekend|holiday hours)\b",
    re.I,
)
_PHONE = re.compile(
    r"\b(phone|call|telephone|number to (?:call|reach)|customer service|contact (?:number|info))\b", re.I
)
_RATE_OR_APR = re.compile(r"\bapr\b|\brate\b")
_ROUTING = re.compile(r"\b(routing|aba)\b", re.I)
_SUPER = re.compile(
    r"\b(highest|best|top|maximum|max|most|lowest|cheapest|minimum rate|least|smallest|biggest|largest)\b", re.I
)
_LOW_SUPER = re.compile(r"(lowest|cheapest|minimum rate|least|smallest)", re.I)
_COMPARE = re.compile(r"\b(more than|over|above|at least|greater than|higher than|exceed\w*|better than|"
                      r"less than|under|below|at most|lower than)\b", re.I)
_LOW_COMPARE = re.compile(r"(less|under|below|at most|lower)", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")
_INTRO = re.compile(r"\b(intro\w*|promo\w*|balance transfer|new member|new cardmember|special offer)\b")
_GENERIC_PRODUCT_WORDS = {"credit", "card", "loan", "certificate", "savings", "checking", "mortgage"}


@dataclass
class Query:
    """Structured form of one question (see module docstring)."""
    raw: str
    # normalised wording, date phrases removed: what FactStore.facts() is given
    text: str
    # value | list | check | change | url | howto | branch | info
    intent: str
    # rate | fee | min_open | min_earn | contact | text
    kind: str
    # wanted attributes, e.g. ('apy',) / ('dividend_rate', 'apy') / ('fee',)
    attrs: tuple
    family: str
    segment: str | None
    qualifiers: dict
    words: list
    # ISO date the answer must hold on (None = latest)
    as_of: str | None
    # 'arg' | 'text' | None
    as_of_source: str | None
    # 'last year' / 'in 2025': used to prefer dated text, never to filter facts
    year_hint: int | None = None
    # percentages / amounts the question quotes (for yes/no checks)
    numbers: list = field(default_factory=list)
    # 'gt' | 'lt' for 'more than 5%'
    compare: str | None = None
    # 'high' | 'low'
    superlative: str | None = None
    wants_intro: bool = False
    asks_phone: bool = False
    asks_routing: bool = False
    # 'in October 2021' on a question about past events (not an as_of): used to prefer dated text
    month_hint: int | None = None
    # set only by the LLM parser (llm/parse.py); the defaults leave the rule-parsed behaviour untouched
    # catalog product the question names
    product: str | None = None
    # 'not the APY'
    exclude_attrs: tuple = ()
    # 'not the jumbo'
    exclude_products: tuple = ()
    # yes | no | uncertain
    answerable: str | None = None
    # advice | off_topic | personal_account_data | not_published_type | nonexistent_product
    reason: str | None = None
    answer_type: str | None = None
    entities: tuple = ()


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


def _wording(text: str) -> str:
    t = text.replace("’", "'")
    t = _NUM_TERM.sub(lambda m: f"{NUMBER_WORDS[m.group(1).lower()]} {m.group(2).lower()}", t)
    for pat, rep in QUESTION_WORDING:
        t = re.sub(pat, rep, t, flags=re.I)
    # a bare 5+ digit number (or a 4-digit one that is not a plausible year) after money-ish words is an amount
    t = re.sub(
        r"(?<![\w$.,/-])(\d{5,9}|(?!(?:19[89]|20[0-3])\d\b)\d{4})(?![\w,.%/-])(?!\s*(?:month|year|day|mo\b|yr\b))",
        lambda m: "$" + m.group(1),
        t,
    )
    return t


def _dates(raw: str, arg: str | None, latest: str):
    """(as_of from the text, year hint, wording with the date phrases removed)."""
    wording = _wording(raw)
    if arg:
        _, hint, stripped = date_from_text(wording, latest or "9999-12-31")
        return None, hint, stripped
    return date_from_text(wording, latest or "9999-12-31", latest)


def _take_negated_attrs(text: str) -> tuple[str, set]:
    """'the rate (not the APY)': the attributes the question rules out, and the text without those phrases."""
    negated = {_NEGATED_ATTR_NAMES.get(m.group(1).lower(), "dividend_rate") for m in _NEGATED_RATE.finditer(text)}
    return _NEGATED_RATE.sub(" ", text), negated


def _classify(low: str, pq, phone: bool, routing: bool) -> tuple[str, tuple]:
    """(kind, wanted attributes) of the question."""
    fee_q = bool(_FEE_WORDS.search(low)) or pq.fee
    rate_q = bool(_RATE_WORDS.search(low))
    if rate_q and _OFFER_CONDITIONS.search(low) and not _RATE_FIGURE.search(low):
        # 'earn the $300 bonus' is an offer's conditions, not a rate
        rate_q = False
    if fee_q and rate_q and not _FEE_NOT_RATE.search(low):
        # 'what APR do you charge' is a rate question
        fee_q = False
    product_ctx = bool(pq.family or pq.words) and not _MEMBERSHIP.search(low)
    if routing:
        return "contact", ("routing_number",)
    if phone and not (rate_q or fee_q):
        return "contact", ("phone",)
    if _MIN_EARN.search(low) and product_ctx:
        return "min_earn", ("min_to_earn",)
    if _MIN_OPEN.search(low) and product_ctx:
        return "min_open", ("min_to_open",)
    if fee_q and not _RATE_NOT_FEE.search(low):
        return "fee", ("fee",)
    if rate_q:
        return "rate", _rate_attrs(low, pq.family)
    if (
        pq.family in _RATE_FAMILIES
        and (_LIST.search(low) or _OFFER_WORDS.search(low))
        and not _NOT_A_RATE_LIST.search(low)
    ):
        # 'what credit cards do you offer' -> products and their rates
        return "rate", _rate_attrs(low, pq.family)
    if pq.family in _RATE_FAMILIES and pq.qualifiers and not _NOT_A_RATE_ROW.search(low):
        # 'money market 50,000 balance', '15 year mortgage'
        return "rate", _rate_attrs(low, pq.family)
    return "text", ()


def parse(question: str, as_of: str | None = None, latest: str = "") -> Query:
    """Parse `question` (see module docstring)."""
    raw = question or ""
    arg = parse_as_of(as_of)
    d_as_of, hint, stripped = _dates(raw, arg, latest)
    final_as_of, source = (arg, "arg") if arg else ((d_as_of, "text") if d_as_of else (None, None))
    stripped, negated = _take_negated_attrs(stripped)
    low = _squash(stripped)
    pq = parse_query(stripped)
    if (
        not pq.family
        # a term of 15+ years with a rate and no product can only be a mortgage
        and (pq.qualifiers.get("term_months") or 0) >= 180
        and re.search(r"\b(rates?|apr|payment|monthly)\b", low)
    ):
        # '30 year rate for under $200,000' can only be a mortgage
        stripped += " mortgage"
        low = _squash(stripped)
        pq = parse_query(stripped)
    phone, routing = bool(_PHONE.search(low)) and not _RATE_OR_APR.search(low), bool(_ROUTING.search(low))
    kind, attrs = _classify(low, pq, phone, routing)
    nums = [float(x) for x in _PCT.findall(low)]
    cm = _COMPARE.search(low)
    comp = ("lt" if _LOW_COMPARE.match(cm.group(1)) else "gt") if cm else None
    sup = _SUPER.search(low)
    superl = ("low" if _LOW_SUPER.match(sup.group(1)) else "high") if sup else None
    intro = bool(_INTRO.search(low))
    intent = _intent(low, kind, bool(nums), pq)
    if negated and kind == "rate":
        attrs = tuple(a for a in attrs if a not in negated) or attrs
    return Query(
        raw,
        " ".join(stripped.split()),
        intent,
        kind,
        attrs,
        pq.family,
        pq.segment,
        dict(pq.qualifiers),
        pq.words,
        final_as_of,
        source,
        hint,
        nums,
        comp,
        superl,
        intro,
        phone and kind == "contact",
        routing,
        exclude_attrs=tuple(sorted(negated)),
    )


def _rate_attrs(low: str, family: str) -> tuple:
    """Which stored attributes answer a rate question."""
    plain = re.search(
        r"\b(?:interest|note|dividend) rates?\b|\brates? (?:and|&|plus|or|,) (?:the )?(?:aprs?|apys?)\b|\b(?:aprs?"
        r"|apys?) (?:and|&|plus|or|,) (?:the )?(?:interest )?rates?\b",
        low,
    )
    if re.search(r"\bapys?\b|annual percentage yield|(?<!high )(?<!high-)\byield\b", low):
        # 'the interest rate and the APY' asks for both figures
        return ("dividend_rate", "apy") if plain else ("apy",)
    if re.search(r"\baprs?\b|annual percentage rate", low):
        return (("rate", "apr") if family == "mortgage" else ("apr", "rate")) if plain else ("apr",)
    if re.search(r"\bdividends?\b", low):
        return ("dividend_rate", "apy")
    if family == "mortgage":
        return ("rate", "apr")
    if family in _DEPOSIT_FAMILIES:
        return ("dividend_rate", "apy", "apr", "rate") if not family else ("dividend_rate", "apy")
    return ("apr", "rate")


def _specific_product(pq) -> bool:
    return bool(pq.words) and any(w not in _GENERIC_PRODUCT_WORDS for w in pq.words)


def _numeric_intent(low: str, has_num: bool, pq) -> str:
    if _CHECK.match(low) and (has_num or _ANY.search(low) or _STILL.search(low) or _COMPARE.search(low)):
        return "check"
    if _LIST.search(low) or (_PLURAL_PRODUCT.search(low) and not pq.qualifiers and not _specific_product(pq)):
        return "list"
    return "value"


def _intent(low: str, kind: str, has_num: bool, pq) -> str:
    if kind == "contact":
        return "value"
    if _LINK_NOUN.search(low) and not _LINK_VERB.search(low):
        return "url"
    if kind in ("rate", "fee") and _WHERE_TO_FIND.search(low):
        return "url"
    if _CHANGE.search(low) and not _NOT_A_CHANGE.search(low) and kind in ("rate", "fee", "min_open"):
        return "change"
    if kind in NUMERIC_KINDS:
        return _numeric_intent(low, has_num, pq)
    unlinked = _LINK_VERB.sub(" ", low)
    if _URLQ.search(unlinked) and _LINK_OR_PAGE.search(unlinked) or _WHEREDO.search(low) and \
            _WHEREDO_ACTION.search(low) and not _BRANCH.search(low.replace("find a", "")):
        return "url"
    if (_LINK_WORD.search(low) and not _LINK_VERB.search(low)) or _LOCATOR.search(low):
        return "url"
    if _BRANCH.search(low):
        return "branch"
    if _HOWTO.search(low):
        return "howto"
    return "info"


# ---- cleaning the raw question: filler removed, 'the X page says' claims reduced, and context split from the question
_FILLER_WORDS = (
    "hi",
    "hello",
    "hey",
    "yo",
    "pls",
    "plz",
    "please",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "rn",
    "right now",
    "quick",
    "quickly",
    "ok",
    "okay",
    "wondering",
    "i was wondering",
    "i wonder",
    "btw",
    "by the way",
    "asap",
    "um",
    "uh",
    "tbh",
    "lol",
    "quick question",
    "real quick",
    "hope you're well",
    "good morning",
    "good afternoon",
)
_FILLER = re.compile(
    r"(?<![\w$%-])(?:"
    + "|".join(sorted((re.escape(w) for w in _FILLER_WORDS), key=len, reverse=True))
    + r")(?![\w$%-])[,!.:;]*",
    re.I,
)
_POSSESSIVE_NAME = re.compile(rf"\b({re.escape(INSTITUTION)}|{re.escape(INSTITUTION)} Credit Union)['’]s\b")

_CLAIM_LEAD = re.compile(
    r"^\W*(?:the\s+)?[\w &/'-]{1,50}?\s+(?:page|site|website|ad|flyer|landing page|brochure)\s+(?:says|said|shows"
    r"|lists|states|advertises|claims|has)\s+",
    re.I,
)
_CLAIM_SAW = re.compile(r"^\W*(?:i|we|my \w+)\s+(?:saw|read|heard|noticed|found|was told)\b", re.I)
_CLAIM_PAGE = re.compile(
    rf"\s+(?:on|from|in|at)\s+(?:the|your|{re.escape(INSTITUTION.lower())}'?s)\s+[\w &/'-]{{1,50}}?\s+(?:page|site|we"
    rf"bsite|ad|flyer|blog post|article|brochure)\b",
    re.I,
)
_CLAIM_ASK = re.compile(
    r"\b(?:is (?:that|this|it)|still|current\w*|right|correct|true|accurate|what you (?:charge|pay|offer)|up to date"
    r"|valid)\b",
    re.I,
)

_FEE_WORD = re.compile(r"\b(?:fees?|charges?|costs?|pay|surcharges?)\b", re.I)
_ABROAD = re.compile(
    r"\b(?:in|from|to|at|while|when)?\s*(?:another|a foreign|other|a different) country\b|\babroad\b|\boverseas\b"
    r"|\binternational (?:purchases?|transactions?|use)\b"
    r"|\boutside (?:of )?the (?:US|U\.S\.|country|states|United States)\b",
    re.I,
)
_WIRE = re.compile(r"\bwire\b", re.I)
_DOMESTIC_WIRE = re.compile(
    r"\b(?:outgoing|within the (?:US|U\.S\.|country|states|United States)|in the (?:US|U\.S\.)|us wire|domestic)\b",
    re.I,
)
_NOT_DOMESTIC_WIRE = re.compile(r"\b(?:international|foreign|incoming|overseas|abroad|receiving|receive)\b", re.I)
# '(not the jumbo)': a negated aside names nothing asked for
_NEGATED_ASIDE = re.compile(r"\s*\((?:not|excluding|except|without)\b[^)]*\)", re.I)
_NEGATED_VARIANT = re.compile(
    r"\bnon[- ]?(?:jumbo|business|personal|promo\w*|special|introductory|intro|student|premium|secured"
    r"|member)\b(?:\s+(?:one|ones|version|kind|type))?",
    re.I,
)
# an age, not a loan term
_AGE = re.compile(r"\b(\d{1,3})[- ]years?[- ]olds?\b", re.I)

_PRONOUN = re.compile(r"\b(?:its|their|it|that|this|those|these|there|them|they)\b", re.I)
_WH_START = re.compile(
    r"^(?:who|what|whats|what's|when|where|which|why|how|do|does|did|can|could|is|are|was|were|will|would|should|may"
    r"|am|i'd like to know|tell me|give me|show me|list)\b",
    re.I,
)


def _claim_without_page(text: str) -> str:
    """'The X page says 3.00% APY. Is that right?' / 'I saw 5.34% on the X page."""
    if not _CLAIM_ASK.search(text) or "?" not in text:
        return text
    if _CLAIM_LEAD.search(text):
        return _CLAIM_LEAD.sub("", text, count=1)
    if _CLAIM_SAW.search(text):
        return _CLAIM_PAGE.sub("", text, count=1)
    return text


def _foreign_fee_wording(text: str) -> str:
    """'fee when I use my card abroad' asks for the foreign transaction fee."""
    return _ABROAD.sub(" foreign transaction ", text) if _FEE_WORD.search(text) else text


def _domestic_wire_wording(text: str) -> str:
    """'outgoing wire within the US' asks for the domestic wire fee."""
    if not (_WIRE.search(text) and _DOMESTIC_WIRE.search(text) and not _NOT_DOMESTIC_WIRE.search(text)):
        return text
    text = re.sub(r"\s+(?:within|in) the (?:US|U\.S\.|country|states|United States)\b", " ", text, flags=re.I)
    return re.sub(
        r"\b(?:outgoing|domestic|us)\s+wire(?: transfers?)?|\bwire(?: transfers?)?\b",
        "domestic wire",
        text,
        count=1,
        flags=re.I,
    )


def clean_question(text: str) -> str:
    cleaned = _FILLER.sub(" ", _POSSESSIVE_NAME.sub(r"\1", text))
    cleaned = _foreign_fee_wording(cleaned)
    cleaned = _domestic_wire_wording(cleaned)
    for pat, rep in ACRONYM_EXPANSIONS:
        cleaned = re.sub(pat, rep, cleaned, flags=re.I)
    cleaned = _NEGATED_ASIDE.sub(" ", cleaned)
    cleaned = _AGE.sub(r"aged \1", cleaned)
    # '(non-jumbo)' names what is NOT asked
    cleaned = _NEGATED_VARIANT.sub(" ", cleaned)
    cleaned = _claim_without_page(cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:-")
    if not re.sub(r"[^A-Za-z0-9]", "", cleaned):
        return text
    return cleaned


def split_context(clean: str):
    pieces = [x.strip() for x in SENT_BREAK_CAPITAL_DIGIT.split(clean) if x.strip()]
    if len(pieces) < 2:
        return "", clean
    k = 0
    while k < len(pieces) - 1 and not pieces[k].endswith("?") and not _WH_START.match(pieces[k]):
        k += 1
    if k == 0:
        return "", clean
    return " ".join(pieces[:k]), " ".join(pieces[k:])


def mentions_pronoun(text: str) -> bool:
    """The sentence refers back to something said before ('is it still ...'): it cannot stand alone."""
    return _PRONOUN.search(text) is not None
