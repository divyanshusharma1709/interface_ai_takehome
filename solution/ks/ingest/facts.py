"""Fact records and what the extractors share: confidences, qualifier matching, the product registry, fact keys."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from ..core.textutil import strip_plural
from .normalize import GENERIC_WORDS, QUAL_RANGE_NAMES, fmt_range, range_contains, tokens


# ---- confidence
class Conf:
    """How far each kind of extracted fact is trusted (0..1); answers use facts at or above ANSWER_MIN."""

    ANSWER_MIN = 0.6
    SUSPECT = 0.5

    # ---- table rows
    TABLE_ROW = 0.9
    TABLE_GENERIC_LABEL = 0.7
    TABLE_UNRESOLVED = 0.4  # no product name could be found at all
    JUNK_TABLE_PENALTY = 0.1
    UNIT_FROM_HEADER_PENALTY = 0.05

    # ---- fee-schedule lines
    FEE_LINE = 0.85
    FEE_LINE_FLATTENED = 0.75  # the PDF text lost its table structure

    # ---- prose
    ROUTING_NUMBER = 0.8  # nine digits after the word 'routing'
    PHONE = 0.5
    PROSE_RATE = 0.5
    PROSE_FEE = 0.5
    PROSE_PERCENT_FEE = 0.7

    # ---- 'label: value' lines and one-rate sentences
    LINE = 0.8  # a short line stating the rate or fee
    SENTENCE = 0.65  # the same inside a longer sentence
    STALE_CEILING = 0.45


# ---- extraction_stats
class ExtractionStats(Counter):
    """Counters (`stats["tables_seen"] += 1`), plus `log`: the tables that yielded nothing, as (kind, url."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[tuple] = []


# ---- fact_qualifiers
CONTACT_ATTRS = {"phone", "routing_number"}  # always offered, whatever their confidence
RATE_ATTRS = {"rate", "apy", "apr", "dividend_rate", "points", "payment_example"}
_RANGE_FOR = {
    "balance": ("balance_range", "amount_range"),
    "loan_amount": ("loan_amount_range", "amount_range"),
    "amount": ("balance_range", "loan_amount_range", "amount_range"),
}
_TIER_NAMES = ("balance_range", "loan_amount_range", "amount_range")
_SAME_NUMBER = 1e-9


def tier_dimension(name: str) -> str:
    """The dimension a qualifier name stands for: every tier kind is 'amount'."""
    return "amount" if name in _RANGE_FOR else name


def qual_match(fq: dict, query: dict, loose: bool) -> bool:
    """Do a fact's qualifiers satisfy a query like {'term_months': 60, 'model_year': 2025, 'balance'."""
    for name, want in query.items():
        if name == "terms":  # several terms asked at once
            have = fq.get("term_months")
            if have is None and loose:
                continue
            if have not in want:
                return False
            continue
        have = fq.get(name)
        if have is None and name in _RANGE_FOR:
            have = next((fq[n] for n in _RANGE_FOR[name] if isinstance(fq.get(n), tuple)), None)
        if have is None:
            if loose:
                continue
            return False
        if not _value_matches(name, have, want):
            return False
    return True


def _value_matches(name: str, have, want) -> bool:
    if isinstance(have, tuple) and isinstance(want, (int, float)):
        lo, hi = have
        if name == "model_year":
            return (lo is None or want >= lo) and (hi is None or want <= hi)
        return range_contains(have, float(want))
    if isinstance(have, tuple) and isinstance(want, tuple):
        return have == want
    if isinstance(have, (int, float)) and isinstance(want, (int, float)):
        return abs(have - want) <= _SAME_NUMBER
    return str(have) == str(want)


def tier_qual(fq: dict):
    """(name, range) of the balance / loan-amount tier a fact is keyed on, or None."""
    return next(((n, fq[n]) for n in _TIER_NAMES if isinstance(fq.get(n), tuple)), None)


def dims(fq: dict) -> set:
    """Dimension names of a qualifier dict, with every tier kind folded into 'amount'."""
    return {("amount" if n.endswith("_range") else n) for n, _ in fq.items() if n not in ("segment", "label", "ext")}


# ---- product_registry
# Words too weak to name a product on their own (or at all) when read out of a sentence.
WEAK_TOKENS = {
    "rate",
    "rates",
    "account",
    "accounts",
    "table",
    "apy",
    "apr",
    "select",
    "an",
    "option",
    "below",
    "and",
    "or",
    "the",
    "of",
    "to",
    "your",
    "monthly",
    "breakdown",
    "deposit",
    "deposits",
    "loans",
    "loan",
    "home",
}
# Words that name a product family rather than one product ('savings' alone is not a product).
FAMILY_TOKENS = {
    "savings",
    "saving",
    "checking",
    "loan",
    "certificate",
    "ira",
    "visa",
    "credit",
    "card",
    "personal",
    "business",
    "term",
    "special",
    "regular",
    "money",
    "market",
    "mortgage",
    "home",
}
# Words that vary a product's name without making it another product ('fixed certificate special' = 'certificate
# special').
TYPE_WORDS = {
    "fixed",
    "variable",
    "regular",
    "standard",
    "basic",
    "traditional",
    "personal",
    "term",
    "special",
    "account",
    "rate",
}
MIN_SINGLE_WORD_NAME = 5


class ProductRegistry:
    """product -> (family, name tokens), in the order the products were first seen."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, frozenset]] = {}
        self._runs: dict[str, list[str]] = {}

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __contains__(self, product: str) -> bool:
        return product in self._entries

    def __iter__(self):
        return iter(self._entries)

    def add(self, product: str, family: str) -> None:
        """Register `product` (the first registration of a name wins)."""
        if product not in self._entries:
            run = tokens(product)
            self._entries[product] = (family, frozenset(run))
            self._runs[product] = run

    def family(self, product: str) -> str:
        return self._entries[product][0]

    def name_tokens(self, product: str) -> frozenset:
        return self._entries[product][1]

    def entries(self):
        """(product, family, name tokens) for every product."""
        return ((p, fam, toks) for p, (fam, toks) in self._entries.items())

    def families(self) -> set:
        return {fam for fam, _ in self._entries.values()}

    def best_in(self, words: list) -> str | None:
        """Longest product whose name words occur consecutively in `words` (a sentence's tokens)."""
        best, best_n = None, 0
        for p, run in self._runs.items():
            if len(run) > best_n and _has_run(words, run):
                if len(run) == 1 and (
                    run[0] in FAMILY_TOKENS or run[0] in WEAK_TOKENS or len(run[0]) < MIN_SINGLE_WORD_NAME
                ):
                    continue
                best, best_n = p, len(run)
        return best

    def match_phrase(self, phrase: str) -> str | None:
        """The product a phrase names: an exact product."""
        pt = [strip_plural(w) for w in tokens(phrase) if w not in WEAK_TOKENS]
        if not pt:
            return None
        exact = [p for p, run in self._runs.items() if [strip_plural(w) for w in run] == pt]
        if exact:
            return sorted(exact)[0]
        if len(pt) < 2:
            return None
        cands = []
        for p, (_, toks) in self._entries.items():
            stems = {strip_plural(w) for w in toks}
            if set(pt) <= stems and stems - set(pt) <= TYPE_WORDS:
                cands.append((len(toks), p))
        cands.sort()
        if not cands or (len(cands) > 1 and cands[0][0] == cands[1][0]):
            return None
        return cands[0][1]


def _has_run(seq: list, run: list) -> bool:
    n = len(run)
    return n > 0 and any(seq[i : i + n] == run for i in range(len(seq) - n + 1))


# ---- facts
__all__ = ["AUTHORITY", "GENERIC_WORDS", "Disagreement", "Fact", "FactKey", "Version", "make_key"]


AUTHORITY = {
    "rate_sheet": 3.0,
    "product": 2.0,
    "branch_contact": 2.0,
    "other": 1.0,
    "newsroom": 1.0,
    "blog": 0.5,
    "form_legal": 0.5,
}


def _fmt_q(k: str, v) -> str:
    if k == "model_year" and isinstance(v, tuple):
        return f"<={v[1]}" if v[0] is None else f"{v[0]}-{v[1]}"
    if k in QUAL_RANGE_NAMES and isinstance(v, tuple):
        return fmt_range(v)
    return str(v)


class FactKey(NamedTuple):
    """Identity of a fact across time and pages."""

    product: str
    attribute: str
    qualifiers: tuple

    @property
    def qdict(self) -> dict:
        return dict(self.qualifiers)

    def __str__(self) -> str:
        qs = ", ".join(f"{k}={_fmt_q(k, v)}" for k, v in self.qualifiers)
        return f"{self.product} | {self.attribute}" + (f" | {qs}" if qs else "")


def make_key(product: str, attribute: str, quals: dict) -> FactKey:
    return FactKey(product, attribute, tuple(sorted((k, v) for k, v in quals.items() if v not in (None, ""))))


@dataclass(eq=False)
class Fact:
    """One value on one page of one snapshot, with the validity window the store works out for it."""

    key: FactKey
    product_raw: str
    family: str
    value: float | None
    unit: str
    source_kind: str
    confidence: float
    value_hi: float | None = None
    raw_value: str = ""
    value_text: str = ""
    raw_url: str = ""
    page_type: str = ""
    page_title: str = ""
    heading: str = ""
    caption: str = ""
    column: str = ""
    row_text: str = ""
    effective_date: str | None = None
    effective_scope: str | None = None
    flags: tuple[str, ...] = ()
    chunk_ordinal: int = 0
    table_index: int = 0
    loose_qualifiers: tuple = ()
    offer_end: str | None = None

    url: str = ""
    snapshot_id: str = ""
    taken_on: str = ""
    fetched_at: str = ""
    first_snapshot: str = ""
    last_snapshot: str = ""
    valid_from: str = ""
    observed_from: str = ""
    valid_until: str | None = None
    ended_reason: str = ""  # 'changed' | 'page_gone' | 'removed' | ''
    alts: list = field(default_factory=list)

    @property
    def product(self) -> str:
        return self.key.product

    @property
    def attribute(self) -> str:
        return self.key.attribute

    @property
    def qualifiers(self) -> dict:
        return self.key.qdict

    @property
    def value_sig(self) -> tuple:
        return (self.value, self.value_hi, self.value_text)

    def display_value(self) -> str:
        if self.value is None:
            return self.value_text
        f = lambda x: f"{x:g}" if self.unit != "$" else f"{x:,.2f}"
        s = f(self.value) + (f"-{f(self.value_hi)}" if self.value_hi is not None else "")
        return s + ("%" if self.unit == "%" else "") if self.unit != "$" else "$" + s.replace("-", "-$")

    @classmethod
    def in_chunk(cls, page, chunk, **values) -> "Fact":
        """An observation read from one chunk of `page`: the page and chunk context is filled in."""
        return cls(
            page_type=page.page_type,
            page_title=page.title,
            raw_url=chunk.source_url.split("#")[0],
            heading=chunk.heading_path,
            chunk_ordinal=chunk.ordinal,
            **values,
        )

    @classmethod
    def on_page(cls, page, **values) -> "Fact":
        """An observation read from the page as a whole (no chunk)."""
        return cls(page_type=page.page_type, page_title=page.title, **values)

    def placed(self, **placement) -> "Fact":
        """A copy of an extracted fact (templates are shared between snapshots) placed in a snapshot."""
        return replace(self, alts=[], **placement)


@dataclass
class Disagreement:
    """Raw material for conflict reporting: a key with several distinct values (see `FactStore.disagreements`)."""

    kind: str
    key: FactKey
    snapshot_id: str
    values: list  # distinct (value, value_hi) pairs
    facts: list
    changes: list = field(default_factory=list)

    def __str__(self) -> str:
        vs = " vs ".join(
            f"{v[0]:g}" + (f"-{v[1]:g}" if v[1] is not None else "") for v in self.values if v[0] is not None
        )
        return f"[{self.kind}] {self.key}: {vs}"


@dataclass
class Version:
    """A run of identical values for one (key, url): the unit of history."""

    key: FactKey
    url: str
    value: float | None
    value_hi: float | None
    value_text: str
    unit: str
    first_snapshot: str
    last_snapshot: str
    valid_from: str
    observed_from: str
    valid_until: str | None
    ended_reason: str
    page_type: str
    effective_date: str | None
    confidence: float
    fact: Fact  # latest observation in the run
    urls: tuple = ()
    facts: list = field(default_factory=list)  # every observation in the run, oldest first
