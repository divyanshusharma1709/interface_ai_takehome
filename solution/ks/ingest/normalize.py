"""Normalisation for fact extraction: values, qualifiers, query parsing, product names, families."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---- product_names
_BUSINESS = re.compile(r"\(?\bbusiness(?:es)?\b\)?", re.I)
_APOS = re.compile(r"[’'`]")
_PHRASES = [
    (r"\bhome equity lines? of credit\b", "heloc"),
    (r"\bcertificates? of deposits?\b", "certificate"),
    (r"\b(?:term )?share certificates?\b", "certificate"),
    (r"\bterm certificates?\b", "certificate"),
    (r"\bcds?\b", "certificate"),
    (r"\bannual percentage yield\b", "apy"),
    (r"\bannual percentage rate\b", "apr"),
    (r"\bboat (?:and|&) rv\b", "boat rv"),
    (r"\bvariable interest rate\b|\binterest rate\b|\bas low as\b", " "),
    (r"\bcredit cards?\b", "credit card"),
]
_SINGULAR = {
    "certificates": "certificate",
    "loans": "loan",
    "iras": "ira",
    "accounts": "account",
    "cards": "card",
    "rates": "rate",
    "months": "month",
    "years": "year",
    "visas": "visa",
    "cds": "certificate",
    "accts": "account",
    "specials": "special",
}
_TRAILING_NOISE = {"rate", "account", "apr", "apy", "rates", "accounts", "table"}
_NOISE = {"standard", "the", "of", "a", "an", "for", "and", "or"}
_STOP = {"and", "the", "a", "an", "of", "for", "to", "in", "on", "with", "or"}

FAMILIES: list[tuple[str, re.Pattern, str]] = [  # (id, pattern, noun appended to a bare name)
    ("heloc", re.compile(r"\bheloc\b"), "heloc"),
    ("home_equity_loan", re.compile(r"\bhome equity\b"), "home equity loan"),
    ("home_improvement_loan", re.compile(r"\bhome improvement\b"), "home improvement loan"),
    (
        "recreational_loan",
        re.compile(r"\b(boat|rv|atv|snowmobile|jet ski|recreational|motorhome)\b"),
        "recreational loan",
    ),
    ("auto_loan", re.compile(r"\b(auto|motorcycle|vehicle|car)\b"), "auto loan"),
    ("personal_loan", re.compile(r"\bpersonal loan\b"), "personal loan"),
    ("credit_card", re.compile(r"\b(visa|credit card|mastercard)\b"), ""),
    ("hsa", re.compile(r"\bhealth savings\b|\bhsa\b"), "health savings"),
    ("ira", re.compile(r"\bira\b|\broth\b|\btraditional\b"), "ira"),
    ("certificate", re.compile(r"\bcertificate\b"), "certificate"),
    ("money_market", re.compile(r"\bmoney market\b"), "money market"),
    ("checking", re.compile(r"\bchecking\b"), "checking"),
    ("savings", re.compile(r"\b(savings?|share|dividend)\b"), "savings"),
    (
        "mortgage",
        re.compile(r"\b(mortgage|conventional|home loan|fha|va loan|jumbo loan|purchase|refinance)\b"),
        "mortgage",
    ),
]
LOAN_FAMILIES = {
    "heloc",
    "home_equity_loan",
    "home_improvement_loan",
    "recreational_loan",
    "auto_loan",
    "personal_loan",
    "credit_card",
    "mortgage",
}
_COLLAPSE = {
    "auto_loan": re.compile(r"^(auto|motorcycle|vehicle|car|loan)( (auto|motorcycle|vehicle|car|loan))*$"),
    "personal_loan": re.compile(r"^(personal|loan)( (personal|loan))*$"),
}


def norm_text(s: str) -> str:
    """Lowercase, accent-free, punctuation-light form used for product names and matching."""
    s = _APOS.sub("", s.lower().replace("&", " and "))
    s = re.sub(r"[–—‑-]", " ", s)
    s = re.sub(r"[^a-z0-9 /]+", " ", s)
    return " ".join(s.split())


def canon_name(s: str) -> str:
    """Normalised product name: synonyms applied, plurals folded, noise and trailing 'rate/account' dropped."""
    s = re.sub(r"\((?:fixed|variable|standard)\)", " ", s, flags=re.I)
    s = norm_text(re.sub(r"\s/\s", " ", s))
    for pat, rep in _PHRASES:
        s = re.sub(pat, rep, s)
    toks = [_SINGULAR.get(w, w) for w in s.split()]
    toks = [w for w in toks if w not in _NOISE]
    while toks and toks[-1] in _TRAILING_NOISE:
        toks.pop()
    out: list[str] = []
    for w in toks:
        if w not in out or w.isdigit():
            out.append(w)
    return " ".join(out)


def strip_segment(name: str) -> tuple[str, bool]:
    """Remove the 'business' marker from a name; returns (name, was_business)."""
    n = _BUSINESS.sub(" ", name)
    return " ".join(n.split()), n != name and bool(_BUSINESS.search(name))


def family_of(*texts: str) -> str:
    """First family whose pattern matches the earliest text that matches any family."""
    for t in texts:
        if not t:
            continue
        n = canon_name(t)
        if not n:
            continue
        for fid, rx, _ in FAMILIES:
            if rx.search(n):
                return fid
    return ""


def family_noun(fid: str) -> str:
    return next((noun for f, _, noun in FAMILIES if f == fid), "")


def has_family_keyword(name: str, fid: str) -> bool:
    return any(rx.search(name) for f, rx, _ in FAMILIES if f == fid)


def finish_product(name: str, fid: str) -> str:
    """Append the family noun to a bare name (e.g."""
    name = " ".join(name.split())
    col = _COLLAPSE.get(fid)
    if col and (not name or col.match(name) or re.fullmatch(r"(auto|motorcycle|vehicle|car|loan|and| )+", name)):
        return family_noun(fid)
    noun = family_noun(fid)
    if noun and not has_family_keyword(name, fid):
        name = f"{name} {noun}".strip()
    return name


def tokens(s: str) -> list[str]:
    """Match tokens for fuzzy product lookup: canonical name words minus stopwords, light stemming."""
    out = []
    for w in canon_name(s).split():
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        out.append(w)
    return out


# ---- normalize
__all__ = ["family_noun", "family_of", "tokens"]  # also re-exported for the answer layer

# ---------------------------------------------------------------- values
_NUM = r"\d[\d,]*(?:\.\d+)?"
_PCT = re.compile(rf"({_NUM})\s*%")
_PCT_RANGE = re.compile(rf"^\W*({_NUM})\s*%\s*(?:APR|APY)?\s*(?:–|—|-|to)\s*({_NUM})\s*%")
_MONEY = re.compile(rf"\$\s*({_NUM})")
_BARE = re.compile(rf"^\W*({_NUM})\s*(?:[A-Za-z]{{0,4}}\s*)?$")


def to_float(s: str) -> float:
    return float(s.replace(",", ""))


@dataclass(frozen=True)
class Value:
    """A parsed numeric cell: `lo` (and `hi` for ranges) in `unit` ('%', '$' or '' for a count)."""

    lo: float
    hi: float | None
    unit: str
    raw: str
    flags: tuple[str, ...] = ()


def parse_value(cell: str, numeric_hint: str = "") -> Value | None:
    """Parse a cell as a percentage, percentage range, money amount or bare number; None for N/A and text."""
    c = cell.strip()
    if not c:
        return None
    m = _PCT_RANGE.match(c)
    if m:
        return Value(to_float(m.group(1)), to_float(m.group(2)), "%", c)
    m = _PCT.search(c)
    if m:
        return Value(to_float(m.group(1)), None, "%", c)
    m = _MONEY.search(c)
    if m:
        return Value(to_float(m.group(1)), None, "$", c)
    if re.fullmatch(r"\W*none\W*", c, re.I) and numeric_hint == "$":
        return Value(0.0, None, "$", c, ("none",))
    if re.fullmatch(r"\W*(free|no charge)\W*", c, re.I) and numeric_hint == "$":
        return Value(0.0, None, "$", c, ("free",))
    m = _BARE.match(c)
    if m and numeric_hint in ("%", "$", "n"):
        return Value(
            to_float(m.group(1)),
            None,
            "%" if numeric_hint == "%" else ("$" if numeric_hint == "$" else ""),
            c,
            ("unit_from_header",),
        )
    return None


# ---------------------------------------------------------------- qualifiers
_YEAR_BAND = re.compile(r"(?:model\s*year\s*)?\b((?:19|20)\d\d)\s*[–—-]\s*((?:19|20)?\d\d)\b(?:\s*model\s*year)?", re.I)
_YEAR_OLDER = re.compile(r"\b((?:19|20)\d\d)\s*model\s*year\s*\((?:and|&)\s*older\)", re.I)
_AMT_RANGE = re.compile(rf"\$\s*({_NUM})\s*(?:–|—|-|to)\s*\$?\s*({_NUM})", re.I)
_AMT_UPTO = re.compile(
    rf"(?:up\s+to|under|less\s+than|below)\s+\$\s*({_NUM})|\$\s*({_NUM})\s*(?:or\s+less|and\s+under|or\s+under)", re.I
)
_AMT_PLUS = re.compile(
    rf"\$\s*({_NUM})\s*(?:\+|or\s+more|and\s+(?:up|over|above)|or\s+greater)|(?:over|above|more\s+than)"
    rf"\s+\$\s*({_NUM})",
    re.I,
)
_TERMS_ENUM = re.compile(
    r"((?:\d{1,3}\s*,\s*)+\d{1,3}\s*,?\s*(?:and|or|&)\s*\d{1,3}|\d{1,3}\s*,\s*\d{1,3}|\d{1,3}\s*(?:and|or|&)"
    r"\s*\d{1,3})\s*[- ]?(months?|mos?|years?|yrs?)\b",
    re.I,
)
_TERM = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(months?|mos?\b\.?|years?|yrs?\b\.?)", re.I)
_ARM = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})\s*(?:adjustable|arm)\b", re.I)
_IRA = re.compile(r"\b(roth|traditional)\b\s*:?", re.I)
_PURPOSE = re.compile(
    r"\b(purchase|refinance|refi|(?:to )?buy(?:ing)? (?:a |an |my |our )?(?:new )?(?:home|house))\b\s*:?", re.I
)

QUAL_RANGE_NAMES = ("balance_range", "loan_amount_range", "amount_range", "model_year")
GENERIC_WORDS = frozenset({"loan", "card", "credit", "account"})


def _tier(lo: float | None, hi: float | None) -> tuple:
    if lo is not None and lo <= 0.01:
        lo = 0.0
    return (lo, hi)


def _year(a: str, b: str) -> tuple[int, int]:
    lo = int(a)
    hi = int(b) if len(b) == 4 else int(a[:2] + b)
    return (min(lo, hi), max(lo, hi))


class _Cutter:
    """A text from which matched phrases are cut out as their qualifiers are read."""

    def __init__(self, text: str):
        self.t = text

    def cut(self, m: re.Match) -> None:
        self.t = self.t[: m.start()] + " " + self.t[m.end() :]

    def take(self, rx: re.Pattern) -> re.Match | None:
        m = rx.search(self.t)
        if m:
            self.cut(m)
        return m


def _lead_qualifiers(c: _Cutter, q: dict) -> None:
    """IRA type, purpose and model year: stated the same way in table cells and in questions."""
    if m := c.take(_IRA):
        q["ira_type"] = m.group(1).lower()
    if m := c.take(_PURPOSE):
        q["purpose"] = "refinance" if m.group(1).lower().startswith("ref") else "purchase"
    if m := c.take(_YEAR_OLDER):
        q["model_year"] = (None, int(m.group(1)))
    elif m := c.take(_YEAR_BAND):
        q["model_year"] = _year(m.group(1), m.group(2))


def _term_qualifiers(c: _Cutter, q: dict, enumerated: bool = False) -> None:
    if m := c.take(_ARM):
        q["arm"] = f"{m.group(1)}/{m.group(2)}"
    if enumerated and (m := c.take(_TERMS_ENUM)):
        q["terms"] = sorted({int(x) * _months(m.group(2)) for x in re.findall(r"\d+", m.group(1))})
    if m := c.take(_TERM):
        q["term_months"] = int(m.group(1)) * _months(m.group(2))


def _months(unit: str) -> int:
    return 12 if unit.lower().startswith("y") else 1


def _amount_tier(c: _Cutter, q: dict) -> None:
    """'$5,000 - $24,999' / 'up to $10,000' / '$50,000+': a balance or loan-amount tier of a table label."""
    if m := c.take(_AMT_RANGE):
        q["amount_range"] = _tier(to_float(m.group(1)), to_float(m.group(2)))
    elif m := c.take(_AMT_UPTO):
        q["amount_range"] = _tier(0.0, to_float(m.group(1) or m.group(2)))
    elif m := c.take(_AMT_PLUS):
        q["amount_range"] = _tier(to_float(m.group(1) or m.group(2)), None)


def _amount_point(c: _Cutter, q: dict) -> None:
    """The one amount a question names ('$15,000', '15k', '15,000'; a range reads as its low end)."""
    if m := c.take(_AMT_RANGE):
        q["amount"] = to_float(m.group(1))
    for rx, value in (
        (_Q_DOLLAR, lambda m: to_float(m.group(1)) * _Q_MULT.get((m.group(2) or "").lower(), 1)),
        (_Q_SUFFIX, lambda m: float(m.group(1)) * _Q_MULT[m.group(2).lower()]),
        (_Q_COMMA, lambda m: to_float(m.group(0))),
    ):
        m = rx.search(c.t)
        if m and "amount" not in q:
            q["amount"] = value(m)
            c.cut(m)


def _model_year(c: _Cutter, q: dict, auto_ctx: bool) -> None:
    c.take(re.compile(r"\bmodel\s*year\b", re.I))
    if "model_year" not in q:
        y = _Q_YEAR_CTX.search(c.t)
        any_year = _Q_YEAR_ANY.search(c.t)
        year = (y.group(1) or y.group(2)) if y else (any_year.group(1) if auto_ctx and any_year else None)
        if year:
            q["model_year"] = int(year)
    c.t = _Q_YEAR_ANY.sub(" ", c.t)


def read_qualifiers(c: _Cutter, question: bool) -> dict:
    """The one qualifier pipeline: IRA type, purpose, model-year band, amount, term(s) are cut out of `c` and returned.
    A table label states an amount as a tier (`amount_range`); a question names one amount, terms and a model year."""
    q: dict = {}
    auto_ctx = question and bool(_Q_AUTOISH.search(c.t))
    _lead_qualifiers(c, q)
    (_amount_point if question else _amount_tier)(c, q)
    _term_qualifiers(c, q, enumerated=question)
    if question:
        _model_year(c, q, auto_ctx)
    return q


def extract_qualifiers(text: str) -> tuple[dict, str]:
    """Pull term / tier / model-year / purpose / IRA-type qualifiers out of a label cell."""
    c = _Cutter(" " + text + " ")
    q = read_qualifiers(c, question=False)
    t = re.sub(r"\(\s*\)|\bmodel\s*year\b|\(and older\)|\bor less\b", " ", c.t, flags=re.I)
    return q, " ".join(t.replace(":", " ").split())


def fmt_range(r: tuple) -> str:
    lo, hi = r
    f = lambda x: f"{x:,.2f}".rstrip("0").rstrip(".") if x is not None else ""
    if lo is None:
        return f"<={f(hi)}"
    if hi is None:
        return f"{f(lo)}+"
    return f"{f(lo)}-{f(hi)}"


def range_contains(r: tuple, x: float) -> bool:
    """True if `x` lies in the tier `r` = (lo, hi); open ends are None."""
    lo, hi = r
    if lo is not None and x < lo - 1e-9:
        return False
    if hi is None:
        return True
    if abs(hi - round(hi)) > 1e-9:
        return x < round(hi + 0.01, 2) - 1e-9
    return x <= hi + 1e-9


def range_width(r: tuple) -> float:
    """Size of a tier; open ends count as infinite so a bounded tier is always 'tighter' than an open one."""
    lo, hi = r
    return float("inf") if hi is None else hi - (0.0 if lo is None else lo)


def tightest(ranges, x: float):
    """The narrowest of the `ranges` containing `x` (None if none do); ties go to the higher lower bound."""
    hits = [r for r in ranges if range_contains(r, x)]
    return min(hits, key=lambda r: (range_width(r), -(r[0] or 0.0))) if hits else None


# ---------------------------------------------------------------- free-text query parsing
_Q_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6}
_Q_DOLLAR = re.compile(rf"\$\s*({_NUM})\s*(k|m|mm|million|thousand)?\b", re.I)
_Q_SUFFIX = re.compile(r"\b(\d+(?:\.\d+)?)\s*(k|million|thousand)\b", re.I)
_Q_COMMA = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d])")
_Q_YEAR_CTX = re.compile(
    r"\b((?:19[89]|20[0-3])\d)\b(?=\s*(?:model|vehicle|car|truck|suv|auto|motorcycle|bike|cycle)|\s*$)|"
    r"\b(?:model|year|a|an|the|for)\s+((?:19[89]|20[0-3])\d)\b",
    re.I,
)
_Q_YEAR_ANY = re.compile(r"\b((?:19[89]|20[0-3])\d)\b")
_Q_AUTOISH = re.compile(r"\b(auto|automobile|vehicle|car|truck|suv|motorcycle|bike|model\s*year|rv|boat)\b", re.I)
_QUERY_ALIASES = [  # generic wording -> the words the site uses
    (r"\bfirst[- ]rate\b", "1st rate"),
    (r"\b(?:motorcycles?|vehicles?|cars?|trucks?|suvs?|automobiles?)\b", "auto"),
    (
        r"\b(?:home|house|housing)(?:\s+(?:purchase|buying|purchasing|refinance|refinancing))?\s+loans?\b|"
        r"\bmortgages?\b",
        "mortgage",
    ),
    (r"\bhome equity (?:lines?(?: of credit)?|credit lines?)\b", "heloc"),
    (r"\bwire transfers?\b", "wire"),
    (r"\bhsas?\b", "health savings"),
    (r"\bindividual retirement (?:accounts?|arrangements?)\b", "ira"),
    (r"\bshare draft\b", "checking"),
    (r"\bhigh[- ]yield\b", "high yield"),
    (r"\bcertificates? of deposits?\b|\bcds?\b|\bshare certificates?\b", "certificate"),
]
QUERY_STOP = frozenset(
    {
        "dupaco",
        "account",
        "accounts",
        "rate",
        "rates",
        "apy",
        "apr",
        "interest",
        "current",
        "currently",
        "what",
        "whats",
        "is",
        "are",
        "was",
        "were",
        "the",
        "my",
        "me",
        "i",
        "you",
        "your",
        "can",
        "could",
        "get",
        "how",
        "much",
        "does",
        "do",
        "cost",
        "costs",
        "charge",
        "charges",
        "for",
        "a",
        "an",
        "on",
        "of",
        "to",
        "in",
        "at",
        "with",
        "per",
        "new",
        "used",
        "tell",
        "about",
        "there",
        "any",
        "which",
        "have",
        "has",
        "offer",
        "offers",
        "offered",
        "payment",
        "if",
        "it",
        "and",
        "or",
        "would",
        "will",
        "be",
        "this",
        "that",
        "from",
        "by",
        "as",
        "than",
        "more",
        "less",
        "over",
        "under",
        "up",
        "minimum",
        "min",
        "balance",
        "deposit",
        "union",
        "community",
        "member",
        "members",
        "standard",
        "dividend",
        "routing",
        "number",
        "point",
        "points",
        "pay",
        "pays",
        "paying",
        "earn",
        "earns",
        "earning",
        "give",
        "gives",
        "need",
        "want",
        "open",
        "opening",
        "start",
        "much",
    }
)


@dataclass
class ParsedQuery:
    """A free-text product question split into qualifier filters and the words that name the product."""

    qualifiers: dict
    words: list
    segment: str | None  # 'business' | 'personal' | None (not stated)
    family: str
    fee: bool  # the question is about a fee
    text: str


def parse_query(text: str) -> ParsedQuery:
    """Parse wording like '84-month auto loan on a 2024 vehicle' or 'Share Savings $15,000' into qualifiers + words."""
    c = _Cutter(" " + (text or "").replace("’", "'") + " ")
    size: list[str] = []
    if m := c.take(re.compile(r"\b(\d{1,3})\s*[x×]\s*(\d{1,3})\b", re.I)):
        size = [m.group(1), "x", m.group(2)]
    q = read_qualifiers(c, question=True)
    low = c.t.lower()
    segment = (
        "business"
        if re.search(r"\bbusiness(?:es)?\b", low)
        else ("personal" if re.search(r"\bpersonal\b(?! loan)", low) else None)
    )
    low = re.sub(r"\bbusiness(?:es)?\b", " ", low)
    for pat, rep in _QUERY_ALIASES:
        low = re.sub(pat, rep, low)
    fee = bool(re.search(r"\b(fees?|charges?|costs?)\b", low))
    fam = family_of(low) or ("mortgage" if "purpose" in q else "")
    words = [w for w in tokens(low) if w not in QUERY_STOP and not w.isdigit() and w not in ("fee", "charge")] + size
    return ParsedQuery(q, words, segment, fam, fee, " ".join(low.split()))
