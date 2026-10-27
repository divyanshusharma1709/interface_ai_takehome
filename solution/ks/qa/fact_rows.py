"""Rows of a fact answer (one row per product, qualifiers and segment) and the answers that are not one value."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from ..core.types import Result, SourceRef
from ..ingest.fact_format import (
    LOAN_FAMILIES_EXCL_CARD,
    authority,
    display_product,
    rank_key,
    url_fit,
)
from ..ingest.facts import GENERIC_WORDS, Fact
from ..ingest.normalize import family_of, range_width, tokens
from ..site_profile import (
    FAMILY_GROUPS,
    FAMILY_HUB_TOPICS,
    FAMILY_LIST_NOUNS,
    INSTITUTION,
    RATES_PAGE_DEFAULT_NOUN,
)
from .fact_pick import FILLER, STEMMED_AUX, FactMatcher, product_text

# ---- Rows of a fact answer: one row per (product, qualifiers, segment), with the value of each asked attribute.
# the tier a row sorts by
SORT_TIER_NAMES = ("balance_range", "loan_amount_range", "amount_range")
# the tier a row is merged / reduced by
TIER_NAMES = ("loan_amount_range", "balance_range", "amount_range")
# a model-year band with no end
_OPEN_END = 9999
_NEW_VEHICLE = re.compile(r"\bnew (?:car|vehicle|auto|truck|suv)s?\b|\bbrand[- ]new\b", re.I)
_USED_VEHICLE = re.compile(r"\bused\b|\bpre-?owned\b", re.I)


@dataclass
class Row:
    product: str
    family: str
    # qualifiers without segment
    quals: dict
    # 'personal' | 'business'
    segment: str
    # attribute -> winning Fact
    vals: dict
    merged_segments: bool = False
    # distinct pages that publish the winning observation
    support: int = 1
    # the last tier merged into this row because it carries the same values
    larger_tiers_to: tuple | None = None
    # Roth and Traditional rows merged into this one
    both_ira_types: bool = False

    def sort_key(self):
        q = self.quals
        tier = next((q[n] for n in SORT_TIER_NAMES if isinstance(q.get(n), tuple)), (0, 0))
        my = q.get("model_year")
        return (
            self.product,
            str(q.get("ira_type", "")),
            str(q.get("purpose", "")),
            q.get("term_months", 0),
            -(my[1] if isinstance(my, tuple) else (my or 0)),
            tier[0] or 0,
            # sorts a tier without an upper bound last
            tier[1] or 1e18,
            str(q.get("phase", "")),
        )

    def tier(self):
        """(name, (low, high)) of the row's amount tier, or None."""
        return next(((n, self.quals[n]) for n in TIER_NAMES if isinstance(self.quals.get(n), tuple)), None)

    def variant_key(self, skip: str | None = None) -> tuple:
        """Rows equal apart from qualifier `skip` are variants of one row."""
        return (self.product, self.segment, tuple(sorted((k, v) for k, v in self.quals.items() if k != skip)),
                self.larger_tiers_to, self.both_ira_types)

    def value_sig(self) -> tuple:
        return tuple(sorted((a, f.value_sig) for a, f in self.vals.items()))


# ---------------------------------------------------------------------------------------------------- building
def _best_observations(pool) -> dict:
    """The winning observation of each (product, qualifiers, attribute)."""
    best: dict[tuple, object] = {}
    for f in pool:
        k = (f.product, f.key.qualifiers, f.attribute)
        if k not in best or rank_key(f) > rank_key(best[k]):
            best[k] = f
    return best


def build_rows(pool) -> list[Row]:
    grouped: dict[tuple, dict] = defaultdict(dict)
    for (prod, quals, attr), f in _best_observations(pool).items():
        qd = dict(quals)
        seg = str(qd.pop("segment", "personal"))
        grouped[(prod, tuple(sorted(qd.items())), seg)][attr] = f
    pages: dict = defaultdict(set)
    for f in pool:
        pages[f.key].add(f.raw_url or f.url)
    rows = [Row(prod, next(iter(v.values())).family, dict(qd), seg, v,
                support=max(len(pages[f.key]) for f in v.values())) for (prod, qd, seg), v in grouped.items()]
    # personal / business rows that carry identical values are one row
    merged: dict[tuple, Row] = {}
    for r in sorted(rows, key=lambda r: r.sort_key()):
        sig = (r.product, tuple(sorted(r.quals.items())), r.value_sig())
        if sig in merged and merged[sig].segment != r.segment:
            merged[sig].merged_segments = True
        else:
            merged.setdefault(sig, r)
    return sorted(merged.values(), key=lambda r: r.sort_key())


# ---------------------------------------------------------------------------------------------------- pruning
def prune_rows(q, rows: list[Row]) -> list[Row]:
    """Headline defaults for dimensions the question left open, then collapse equal-valued twin rows."""
    if not rows:
        return rows
    rows = _asked_purpose(q, rows)
    if q.kind == "fee":
        rows = _one_row_per_fee(rows)
    rows = _collapse_ira_types(rows)
    rows = _one_model_window(q, rows)
    rows = _distinct_model_bands(q, rows)
    rows = _tiers(q, rows)
    # ... for a 'list all' question
    # rows an answer lists
    limit = 24 if q.intent == "list" else 12
    if len(rows) > limit and not isinstance(q.qualifiers.get("model_year"), (int, tuple)):
        rows = _newest_band_only(rows)
    if len(rows) > limit:
        rows = _lowest_tier(rows)
    return rows[:limit]


def _asked_purpose(q, rows: list[Row]) -> list[Row]:
    """Purchase over refinance unless the question says which."""
    if not q.qualifiers.get("purpose") and any(r.quals.get("purpose") == "purchase" for r in rows):
        rows = [r for r in rows if r.quals.get("purpose", "purchase") == "purchase"]
    if q.qualifiers.get("purpose"):
        rows = [r for r in rows if r.quals.get("purpose") in (None, q.qualifiers["purpose"])]
    return rows


def _one_row_per_fee(rows: list[Row]) -> list[Row]:
    """One row per fee name and amount, best-sourced first."""
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: -max(f.confidence for f in r.vals.values())):
        sig = (r.product, r.value_sig())
        if sig not in seen:
            seen.add(sig)
            uniq.append(r)
    return sorted(uniq, key=lambda r: r.sort_key())


def _newest_band_only(rows: list[Row]) -> list[Row]:
    years = {r.quals["model_year"] for r in rows if isinstance(r.quals.get("model_year"), tuple)}
    if not years:
        return rows
    newest = max(years, key=lambda y: y[1])
    return [r for r in rows if r.quals.get("model_year") in (None, newest)]


def _collapse_ira_types(rows: list[Row]) -> list[Row]:
    """Rows that differ only in the IRA type (Roth / Traditional) and carry equal values become one row."""
    out: dict = {}
    for r in rows:
        if "ira_type" not in r.quals:
            out[id(r)] = r
            continue
        sig = (
            r.product,
            tuple(sorted((k, v) for k, v in r.quals.items() if k != "ira_type")),
            r.segment,
            r.value_sig(),
        )
        if sig in out:
            out[sig].quals.pop("ira_type", None)
            out[sig].both_ira_types = True
        else:
            out[sig] = r
    return sorted(out.values(), key=lambda r: r.sort_key())


def _band(r: Row) -> tuple:
    lo, hi = r.quals["model_year"]
    return lo or 0, hi or _OPEN_END


def _one_model_window(q, rows: list[Row]) -> list[Row]:
    year = q.qualifiers.get("model_year")
    if not isinstance(year, int):
        return rows
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        my = r.quals.get("model_year")
        if isinstance(my, tuple) and (my[0] or 0) <= year <= (my[1] or _OPEN_END):
            groups[r.variant_key("model_year")].append(r)
    drop = set()
    for rs in groups.values():
        if len(rs) > 1:
            best = max(rs, key=lambda r: (r.support, r.quals["model_year"][1] or _OPEN_END))
            drop |= {id(r) for r in rs if r is not best}
    return [r for r in rows if id(r) not in drop]


def _distinct_model_bands(q, rows: list[Row]) -> list[Row]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        if isinstance(r.quals.get("model_year"), tuple):
            groups[r.variant_key("model_year")].append(r)
    newest_only = q.intent in ("value", "check") or bool(_NEW_VEHICLE.search(q.raw)) and not _USED_VEHICLE.search(q.raw)
    drop = set()
    for rs in groups.values():
        kept: list[Row] = []
        for r in sorted(rs, key=lambda r: (-r.support, -(r.quals["model_year"][1] or 0))):
            lo, hi = _band(r)
            # model-year bands overlapping by this many years are one row seen on two pages
            if any(min(hi, _band(k)[1]) - max(lo, _band(k)[0]) >= 2 for k in kept):
                drop.add(id(r))
            else:
                kept.append(r)
        if newest_only and kept:
            top = max(kept, key=lambda r: r.quals["model_year"][1] or 0)
            drop |= {id(r) for r in kept if r is not top}
    return [r for r in rows if id(r) not in drop]


def _by_tier_family(rows: list[Row]) -> dict:
    """Rows grouped by everything but their amount tier."""
    by: dict[tuple, list[Row]] = defaultdict(list)
    for r in rows:
        t = r.tier()
        by[r.variant_key(t[0] if t else None)].append(r)
    return by


def _tier_start(r: Row) -> tuple:
    low, high = (r.tier() or ("", (0, 0)))[1]
    return low or 0, range_width((low, high))


def _lowest_tier(rows: list[Row]) -> list[Row]:
    out = []
    for grp in _by_tier_family(rows).values():
        grp.sort(key=_tier_start)
        out.append(grp[0])
    return sorted(out, key=lambda r: r.sort_key())


def _tiers(q, rows: list[Row]) -> list[Row]:
    """Balance / loan-amount tiers the question did not fix."""
    if q.qualifiers.get("amount") is not None:
        return rows
    if q.as_of and any(r.family in LOAN_FAMILIES_EXCL_CARD for r in rows):
        return _lowest_tier(rows)
    out = []
    for grp in _by_tier_family(rows).values():
        if len(grp) == 1 or grp[0].tier() is None:
            out.extend(grp)
            continue
        grp.sort(key=_tier_start)
        last_sig = None
        for r in grp:
            if r.value_sig() == last_sig:
                out[-1].larger_tiers_to = r.tier()[1]
                continue
            last_sig = r.value_sig()
            out.append(r)
    return sorted(out, key=lambda r: r.sort_key())

# ---- Fact answers that are not one value: the products of a family, the page that publishes rates, and a contact line.
_LISTY = frozenset(
    {
        "kind",
        "kinds",
        "type",
        "types",
        "account",
        "accounts",
        "product",
        "products",
        "option",
        "options",
        "offer",
        "offers",
        "have",
        "available",
        INSTITUTION.lower(),
        "different",
        "all",
        "get",
        "use",
        "personal",
        "consumer",
        "list",
        "choose",
        "pick",
        "apply",
        "choice",
        "choices",
        "variety",
    }
)
_SPECIFIC_PRODUCT = re.compile(r"\bterms?\b|\bregular\b|\bspecial\b|\bjumbo\b", re.I)
_MIN_CONFIDENCE = 0.6
# 5-letter stems every product shares
_GENERIC_STEMS = {
    "savin",
    "accou",
    "check",
    "card",
    "cards",
    "credi",
    "visa",
    "busin",
    "perso",
    INSTITUTION.lower()[:5],
}
_STEM = 5
# a page that publishes rates as its own business (a rate sheet or product page)
_PAGE_AUTHORITY = 2.0
_INTRO_BALANCE_TRANSFER = "introductory balance transfer"
_RATES_HUB_URL = re.compile(r"/(?:business-)?rates?/?$")
_LOCATOR_WORDS = frozenset(
    {
        "where",
        "see",
        "find",
        "page",
        "pages",
        "link",
        "url",
        "website",
        "site",
        "all",
        "view",
        "look",
        "check",
        "show",
        "access",
        "loan",
        "loans",
        "deposit",
        "deposits",
        "current",
        "latest",
        "published",
        "list",
        "online",
        "web",
        "personal",
        "business",
        "consumer",
        "available",
        "posted",
        "listed",
    }
)
_LOCATOR_FILLER = re.compile(
    r"\b(link|links|url|urls|use|should|can|could|do|does|i|me|where|find|see|view|look|check|page|pages|"
    r"website|site|web|get|give|share|need|want|to|for|the|a|an|of|is|are|on|at|and|our|your|my)\b",
    re.I,
)
_CONTACT_LABELS = {"phone": "phone number", "routing_number": "routing number"}

# (query, family topic words, business?) -> (hub page, product headings)
HubFinder = Callable[[object, set, bool], tuple]


def family_noun(family: str) -> str:
    return FAMILY_LIST_NOUNS[family][0] if family in FAMILY_LIST_NOUNS else family.replace("_", " ")


class FactLists:
    def __init__(self, kb, matcher: FactMatcher, hub: HubFinder | None = None):
        self.kb = kb
        self.fs = kb.facts
        self.matcher = matcher
        self.hub = hub

    # ------------------------------------------------------------------ products of a family
    def product_list(self, q) -> Result | None:
        fams = FAMILY_GROUPS.get(q.family)
        if not fams:
            return None
        # 'money', 'market' name the family itself
        listy = _LISTY | set(tokens(q.family.replace("_", " ")))
        if _SPECIFIC_PRODUCT.search(q.raw) or any(
            w not in listy and w not in FILLER and not family_of(w) and w not in GENERIC_WORDS and w not in STEMMED_AUX
            for w in q.words
        ):
            # a specific product or term was asked for
            return None
        seg = q.segment or "personal"
        names = self._rate_products(q, fams, seg)
        topic = FAMILY_HUB_TOPICS.get(q.family)
        hub_page, heads = self.hub(q, topic, seg == "business") if (self.hub and topic) else (None, [])
        if heads:
            return self._hub_list(q, seg, hub_page, heads, names)
        return self._rate_sheet_list(q, fams, names) if names else None

    def _rate_products(self, q, fams, seg: str) -> dict[str, Fact]:
        """The best-ranked rate fact of each current product of the families, by product name."""
        names: dict[str, Fact] = {}
        for f in self.fs.view(q.as_of):
            if (
                f.family in fams
                and f.attribute in ("apy", "apr", "rate", "dividend_rate")
                and f.confidence >= _MIN_CONFIDENCE
                and f.qualifiers.get("segment", "personal") == seg
                and not f.qualifiers.get("phase")
                and not f.product.endswith(_INTRO_BALANCE_TRANSFER)
            ):
                cur = names.get(f.product)
                if cur is None or rank_key(f) > rank_key(cur):
                    names[f.product] = f
        return names

    def _hub_list(self, q, seg: str, hub_page, heads: list[str], names: dict[str, Fact]) -> Result:
        """The family's hub page names every current product; rate-sheet names it lacks are appended."""
        cover = {x[:_STEM] for h in heads for x in tokens(h) if x[:_STEM] not in _GENERIC_STEMS}
        extras = sorted(
            {
                display_product(f.product, f.family)
                for f in names.values()
                if f.family == q.family
                and not ({x[:_STEM] for x in tokens(f.product) if x[:_STEM] not in _GENERIC_STEMS} & cover)
            }
        )
        items = list(dict.fromkeys(heads + extras))
        srcs = [SourceRef(hub_page.raw_url, hub_page.snapshot_id, "; ".join(heads)[:200])]
        business = seg + " " if seg == "business" else ""
        return Result(
            f"{self.kb.voice.poss()} {business}{family_noun(q.family)} products: " + "; ".join(items) + ".",
            srcs,
            True,
            None,
        )

    def _rate_sheet_list(self, q, fams, names: dict[str, Fact]) -> Result:
        order = {fam: i for i, fam in enumerate(fams)}
        items = sorted(names.values(), key=lambda f: (order.get(f.family, 9), f.product))
        text = (
            f"{self.kb.voice.poss()} {family_noun(q.family)} products: "
            + "; ".join(display_product(f.product, f.family) for f in items)
            + "."
        )
        urls = []
        for f in items:
            u = f.raw_url or f.url
            if u not in urls:
                urls.append(u)
        return Result(
            text,
            [SourceRef(u, next(g.snapshot_id for g in items if (g.raw_url or g.url) == u), None) for u in urls[:4]],
            True,
            None,
        )

    # ------------------------------------------------------------------ where are the rates
    def url_answer(self, q) -> Result | None:
        """URL question about rates: matched facts' pages, or every rate-bearing page when no product is named."""
        words = [
            w
            for w in q.words
            if w not in FILLER and not family_of(w) and w not in GENERIC_WORDS and w not in _LOCATOR_WORDS
        ]
        if q.family or words:
            text = _LOCATOR_FILLER.sub(" ", product_text(q))
            pool = self.matcher.filter(q, self.fs.facts(" ".join(text.split()), as_of=q.as_of, segment="auto"))
        else:
            seg = q.segment or "personal"
            pool = [
                f
                for f in self.fs.view(q.as_of)
                if f.attribute in ("apy", "apr", "rate")
                and f.confidence >= _MIN_CONFIDENCE
                and f.qualifiers.get("segment", "personal") == seg
                and authority(f) >= _PAGE_AUTHORITY
            ]
        return self._rate_page(q, pool) if pool else None

    @staticmethod
    def _rate_page(q, pool: list[Fact]) -> Result | None:
        """'Where can I see the CD rates?': the pages that publish the matched facts, product pages before the hub."""
        pages: dict[str, list[Fact]] = defaultdict(list)
        for f in pool:
            pages[f.raw_url or f.url].append(f)
        if not pages:
            return None
        ranked = sorted(pages, key=lambda u: (-url_fit(u, pages[u][0].family), -len(pages[u]), u))
        good = [
            u for u in ranked if authority(pages[u][0]) >= _PAGE_AUTHORITY and url_fit(u, pages[u][0].family) > 0
        ] or ranked
        sheets = [u for u in good if authority(pages[u][0]) >= 3.0]
        hub = [u for u in good if _RATES_HUB_URL.search(u)]
        urls = (hub if (hub and not q.family) else sheets if (sheets and not q.family) else good)[:3]
        noun = FAMILY_LIST_NOUNS[q.family][1] if q.family in FAMILY_LIST_NOUNS else RATES_PAGE_DEFAULT_NOUN
        text = f"Current {noun} rates are published at " + " and ".join(urls) + "."
        return Result(
            text, [SourceRef(u, pages[u][0].snapshot_id, (pages[u][0].row_text or "")[:200]) for u in urls], True, None
        )

    # ------------------------------------------------------------------ contact facts
    def contact(self, q) -> Result | None:
        """The institution's phone or routing number: the value most pages agree on."""
        attr = q.attrs[0]
        pool = [f for f in self.fs.view(q.as_of) if f.attribute == attr]
        if not pool:
            return None

        def show(g):
            """Digits of a routing number (value or text), or the phone text."""
            return g.value_text if attr == "phone" or g.value is None else str(int(g.value))

        top = Counter(show(f) for f in pool).most_common(1)[0][0]
        same = sorted([g for g in pool if show(g) == top], key=rank_key, reverse=True)
        seen, srcs = set(), []
        # every page that publishes the number, best first
        for g in same:
            u = g.raw_url or g.url
            if u not in seen and len(srcs) < 6:
                seen.add(u)
                srcs.append(SourceRef(u, g.snapshot_id, (g.row_text or "")[:200]))
        return Result(f"{self.kb.voice.poss()} {_CONTACT_LABELS[attr]} is {show(same[0])}.", srcs, True, None)

