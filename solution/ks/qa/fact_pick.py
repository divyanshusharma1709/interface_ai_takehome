"""Which facts answer a question: the asked attributes, intro-rate rules, tier and amount choice."""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import cached_property

from ..ingest.facts import GENERIC_WORDS, Fact
from ..ingest.normalize import QUERY_STOP, family_of, tokens
from ..site_profile import INSTITUTION, strip_title_suffix

# generic words that may remain in a rate question without naming a product
FILLER = frozenset(
    """best good great low lowest high highest top cheap cheapest better popular typical average general regular
right now today currently current latest right new old older newer please tell show give list all every each
options option available offered offer offers rates rate fees fee interest earn earned earning pay pays paid
paying get getting make making would could should might like looking want need know find see look check
compare comparison per yearly annual monthly total whole full model models year years vehicle vehicles car
cars just opened open opening started recently already""".split()
)
# verbs and purposes that surround a rate question without naming a product ('what APR do you charge on a purchase
# loan')
RATE_VERBS = frozenset(
    """charge charges charged cost costs quote quotes quoted give gives gave have has pay pays paid purchase purchases
purchasing buy buying bought home house price priced pricing""".split()
)
FEE_FILLER = frozenset(
    """when made while purchase purchases purchased rent rental renting price prices pricing priced order obtain
request issue send use sending using make take put keep hold buy sell cost costs charge charged charges much
many transaction transactions per each month monthly annual yearly one two three item items service services""".split()
)
# words that decorate any product name without naming a variant ('6-month term certificate', 'regular savings account')
GENERIC_MODIFIERS = (
    frozenset(
        """term terms account accounts plan plans product products option options standard regular basic traditional
typical normal usual general main primary current new existing fixed-term specific particular certain
individual single personal consumer member members credit union cash back""".split()
    )
    | {INSTITUTION.lower()}
)
# 'does', 'was', 'has', 'this', 'are' after light stemming
STEMMED_AUX = frozenset({"doe", "wa", "ha", "thi", "ar"})

# figures a row shows together
PAIRED = {"rate": "apr", "apr": "rate", "dividend_rate": "apy", "apy": "dividend_rate"}
# when the asked one is not published
_FALLBACK_ATTRS = {
    "apy": ("dividend_rate", "apr", "rate"),
    "apr": ("rate", "apy"),
    "dividend_rate": ("apy",),
    "rate": ("apr",),
}
_INTRO_BALANCE_TRANSFER = "introductory balance transfer"
_LOCATOR_FAMILY_NOUNS = ("loan", "loans", "account", "accounts", "card", "cards", "certificate")
_EXTRA_PRODUCT_WORDS = {
    "purchase",
    "refinance",
    "refi",
    "roth",
    "traditional",
    "business",
    "personal",
    "month",
    "months",
    "year",
    "years",
}
_ASKING_VERBS = {"you", "your", "do", "does", "did"}
_FILLER_QUESTION_WORDS = re.compile(
    r"\b(list|each|every|all|one|ones|options?|available|offered?|types?|kinds?|which|what|show|give|get|me|"
    r"did|does|do|has|have|had|changed?|changing|go|goes|went|gone|up|down|increased?|decreased?|dropped?|risen|rise|"
    r"still|anymore|any more|recently|lately|since|now|ever|"
    r"terms?|choices?|please|tell|current|currently|right now|today)\b",
    re.I,
)
_RATE_VERB = re.compile(r"\b(?:charg\w*|costs?|quot\w*|pays?|paying)\b", re.I)
_RATE_WORDS = re.compile(
    r"\b(?:aprs?|apys?|annual percentage (?:rate|yield)|interest rates?|dividend rates?|dividends?)\b", re.I
)
_APR_OR_APY = re.compile(r"\bapr\b|\bapy\b", re.I)
_TIERS_ASK = re.compile(r"\btiers?\b", re.I)


def product_text(q) -> str:
    """The normalised question minus list/filler wording, which would otherwise count as unknown product words."""
    t = " " + re.sub(r"['’]s\b", " ", q.text) + " "
    t = _FILLER_QUESTION_WORDS.sub(" ", t)
    # 'what APR do you charge / quote / pay': the verb is not a fee or a product word
    if q.kind == "rate":
        t = _RATE_VERB.sub(" ", t)
    if q.kind == "rate" and len(q.attrs) > 1 and _APR_OR_APY.search(t):
        # both figures asked: the store must not pick one by its word
        t = _RATE_WORDS.sub(" ", t)
    return " ".join(t.split())


class FactMatcher:
    """Chooses among the facts the store matched to a question."""

    def __init__(self, kb):
        self.kb = kb
        self.fs = kb.facts

    # ------------------------------------------------------------------ the store's product index
    # FactStore offers its product index only through private attributes; the methods below (and `product_vocab`) are
    # the only readers of them.
    def ready(self) -> None:
        """Builds the store's indexes (a no-op when they are current)."""
        self.fs._build()

    def product_words(self) -> dict:
        """Product name -> the words that name it (and its family)."""
        return self.fs._pfull

    def has_product(self, name: str) -> bool:
        return name in self.fs._pfull

    @cached_property
    def product_vocab(self) -> set:
        """Every word of every product name in the store."""
        vocab: set = set()
        for p in list(self.fs._pfull) + list(self.fs._pfamily):
            vocab |= set(tokens(p))
        return vocab

    @cached_property
    def title_vocab(self) -> set:
        """Words that name a product or service on the site."""
        v: set = set()
        view = self.kb.views[-1] if self.kb.views else None
        for page in (view.pages.values() if view else ()):
            if page.page_type in ("product", "rate_sheet") and not page.parsed.listing and page.parsed.fmt == "html":
                v |= set(tokens(strip_title_suffix(page.title or "")))
        return v

    def known_words_only(self, text: str) -> str:
        """`text` without the words no product or fee name contains ('replace', 'lost')."""
        noise = (QUERY_STOP | FILLER | FEE_FILLER | _ASKING_VERBS) - {"fee", "fees", "rate", "rates"}
        keep = [
            w
            for w in text.split()
            if w.lower().strip("?.,!") not in noise
            and (
                any(t in self.product_vocab or t in _EXTRA_PRODUCT_WORDS for t in tokens(w))
                or any(c.isdigit() for c in w)
            )
        ]
        return " ".join(keep)

    # ------------------------------------------------------------------ attributes
    @staticmethod
    def _asked_attributes(q, cands: list[Fact]) -> list[Fact]:
        wanted = list(q.attrs)
        by_attr = [f for f in cands if f.attribute in wanted]
        if by_attr and q.kind == "rate" and q.intent != "check":
            # a rate row is always shown whole (rate + APR, dividend rate + APY): the same row must read the same
            # whichever of its paired figures the wording asked for
            keys = {(f.product, f.key.qualifiers) for f in by_attr}
            partners = {PAIRED[a] for a in wanted if a in PAIRED}
            by_attr = by_attr + [
                f
                for f in cands
                if f.attribute in partners and f.attribute not in wanted and (f.product, f.key.qualifiers) in keys
            ]
        if not by_attr and q.kind == "rate":
            for w in wanted:
                by_attr = [f for f in cands if f.attribute in _FALLBACK_ATTRS.get(w, ())]
                if by_attr:
                    break
        return by_attr

    def filter(self, q, cands: list[Fact]) -> list[Fact]:
        """The facts of the asked attributes, without introductory rates unless asked and without unreadable values."""
        # 'not the jumbo' (LLM parser)
        if q.exclude_products:
            gone = {p.lower() for p in q.exclude_products}
            cands = [f for f in cands if f.product.lower() not in gone]
        by_attr = self._asked_attributes(q, cands)
        out = []
        for f in by_attr:
            ph = str(f.qualifiers.get("phase", ""))
            if (ph.startswith("intro") or f.product.endswith(_INTRO_BALANCE_TRANSFER)) and not q.wants_intro:
                continue
            if (
                q.wants_intro
                and f.family == "credit_card"
                and not (ph.startswith("intro") or "introductory" in f.product)
            ):
                continue
            if "ocr_suspect" in f.flags or f.value is None:
                continue
            out.append(f)
        if q.wants_intro and not any(f.qualifiers.get("phase") or "introductory" in f.product for f in out):
            return [f for f in by_attr if f.value is not None]
        return out

    # ------------------------------------------------------------------ narrowing the pool
    @staticmethod
    def widest_tiered_product(pool: list[Fact], q) -> list[Fact]:
        if not _TIERS_ASK.search(q.raw):
            return pool
        tiered = [f for f in pool if isinstance(f.qualifiers.get("balance_range"), tuple)]
        if not tiered:
            return pool
        seen: dict = {}
        for f in tiered:
            seen.setdefault(f.product, set()).add(f.qualifiers["balance_range"])
        top = max(len(v) for v in seen.values())
        widest = [p for p in seen if len(seen[p]) == top]
        base = min(widest, key=lambda p: (len(tokens(p)), p))
        return [f for f in tiered if f.product == base]

    def by_amount(self, q, pool: list[Fact]) -> list[Fact]:
        """Sibling products priced by size (regular vs jumbo certificate)."""
        amount = q.qualifiers.get("amount")
        prods = {f.product for f in pool}
        if not isinstance(amount, (int, float)) or len(prods) < 2:
            return pool
        floors: dict[str, float] = {}
        for f in self.fs.view(q.as_of):
            if f.product in prods and f.attribute in ("min_to_earn", "min_to_open") and f.value is not None:
                floors[f.product] = min(floors.get(f.product, f.value), f.value)
        if len(floors) < 2:
            return pool
        toks = {p: set(tokens(p)) for p in floors}
        siblings = [p for p in floors if any(o != p and (toks[o] < toks[p] or toks[p] < toks[o]) for o in floors)]
        if len(siblings) < 2:
            return pool
        meets = [p for p in siblings if floors[p] <= amount]
        pick = max(meets, key=lambda p: floors[p]) if meets else min(siblings, key=lambda p: floors[p])
        return [f for f in pool if f.product == pick or f.product not in siblings]

    def pick_products(self, q, pool: list[Fact]) -> list[Fact]:
        """Drop 'extended' products (jumbo, business-only name additions) unless the question names them."""
        prods = {f.product for f in pool}
        if len(prods) < 2 or q.intent == "list" or q.superlative or (q.intent == "check" and q.compare):
            return pool
        words = set(tokens(" ".join(q.words)))
        pf = self.product_words()
        df = Counter(w for p in pf for w in pf[p])
        weight = {w: math.log(1 + len(pf) / df[w]) for w in words if w in df}
        score = {p: sum(weight[w] for w in weight if w in pf.get(p, ())) for p in prods}
        top = max(score.values())
        prods = {p for p in prods if score[p] >= top - 1e-9}
        pool = [f for f in pool if f.product in prods]
        drop = set()
        for p in prods:
            tp = set(tokens(p))
            for o in prods:
                if o != p and set(tokens(o)) < tp and not ((tp - set(tokens(o))) & words):
                    drop.add(p)
        keep = [f for f in pool if f.product not in drop]
        return keep or pool

    # ------------------------------------------------------------------ question words nothing explains
    def unexplained(self, q, pool: list[Fact]) -> list[str]:
        """Question words that the matched products do not explain."""
        known: set = set()
        for f in pool:
            known |= (
                set(tokens(f.product))
                | set(tokens(f.row_text or ""))
                | set(tokens(" ".join(str(v) for v in f.qualifiers.values())))
            )
        bad = []
        for w in q.words:
            if (
                w in known
                or w in QUERY_STOP
                or w.isdigit()
                or len(w) <= 2
                or family_of(w)
                or w in GENERIC_WORDS
                or w in STEMMED_AUX
            ):
                continue
            if q.kind == "fee":
                if not self._explained_fee_word(w, known):
                    bad.append(w)
            elif w in self.title_vocab and w not in FILLER and w not in RATE_VERBS:
                bad.append(w)
        if q.kind != "fee":
            bad += self._type_words_before_family(q, known, bad)
        return bad

    @staticmethod
    def _explained_fee_word(w: str, known: set) -> bool:
        stem = re.sub(r"(?:ing|ed)$", "", w)
        # 'copies' ~ 'copy'
        alt = w[:-2] + "y" if w.endswith("ie") else w
        if alt in known:
            return True
        return (
            w in FILLER
            or w in FEE_FILLER
            or w in GENERIC_MODIFIERS
            or (
                w.endswith(("ing", "ed"))
                and (stem in FEE_FILLER or stem + "e" in FEE_FILLER or stem in FILLER or len(stem) <= 3)
            )
        )

    def _type_words_before_family(self, q, known: set, bad: list[str]) -> list[str]:
        """A type word in front of the family noun ('balloon mortgage') that no product carries."""
        found: list[str] = []
        # LLM-structured: the canonical wording
        words = re.findall(r"[a-z][a-z-]+", (q.text if q.answerable == "yes" else q.raw).lower())
        vocab = self.product_vocab
        for prev, nxt in zip(words, words[1:]):
            prev = prev.split("-")[-1]
            t = tokens(prev)
            if not t or not (family_of(nxt) or nxt in _LOCATOR_FAMILY_NOUNS) or prev in bad or prev in found:
                continue
            if (
                prev in GENERIC_MODIFIERS
                or t[0] in vocab
                or t[0] in known
                or prev in QUERY_STOP
                or prev in FILLER
                or prev in GENERIC_WORDS
                or t[0] in STEMMED_AUX
                or len(prev) <= 2
                or family_of(prev)
            ):
                continue
            found.append(prev)
        return found
