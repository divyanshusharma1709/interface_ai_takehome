"""The query parser: the catalog the prompt offers, validation and merging of the model's reply, and `LLMParser`."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import date
from functools import cached_property

from ..ingest.normalize import parse_query
from ..qa.branches import BranchTable
from ..site_profile import FAMILY_HINTS, FAMILY_NOUNS, ORG_WORDS_SITE
from .client import SOLUTION_DIR, Budget, LLMSettings, Session, normalize_question
from .prompts import PARSER_OMIT, PARSER_SYSTEM

# ---- catalog
# words that carry no identity in a product name: dropped when deriving its short form (and when matching a name to
# the question)
SHORT_NAME_DROP = {
    "fee",
    "fees",
    "special",
    "account",
    "accounts",
    "the",
    "savings",
    "certificate",
    "checking",
    "loan",
    "money",
    "market",
    "visa",
    "ira",
}


def build_catalog(kb) -> dict:
    """{'families': [...], 'products': {lower name: (family, name)}, 'branches'."""
    fs = kb.facts
    fs._build()
    products: dict[str, tuple[str, str]] = {}
    families = set()
    for f in fs.view(None):
        if f.family:
            families.add(f.family)
        if f.product and f.family:
            products.setdefault(f.product.lower(), (f.family, f.product))
    return {
        "families": sorted(families),
        "products": products,
        "branches": branch_names(kb),
        "aliases": _aliases(products),
    }


def branch_names(kb) -> list[str]:
    """Branch names from the newest snapshot; empty when that snapshot cannot be read as a branch table."""
    try:
        if kb.views:
            return sorted({b.name for b in BranchTable(kb.views[-1]).branches.values() if b.name})
    except Exception:
        pass
    return []


def _aliases(products: dict) -> dict:
    seen: dict[str, list] = {}
    for full in products:
        short = " ".join(w for w in full.split() if w not in SHORT_NAME_DROP and not w.isdigit()) or full
        seen.setdefault(short, []).append(full)
    return {s: v[0] for s, v in seen.items() if len(v) == 1 and s not in products}


def catalog_digest(cat: dict) -> str:
    """Part of the cache key: a different catalog is a different prompt."""
    h = hashlib.sha256(json.dumps([cat["families"], sorted(cat["products"]), cat["branches"]]).encode()).hexdigest()
    return h[:10]


# ---- merge
_KIND_ATTRS = {"fee": ("fee",), "min_open": ("min_to_open",), "min_earn": ("min_to_earn",), "text": ()}
_RATE_ATTRS = ("apy", "apr", "dividend_rate", "rate")
_ATTR_WORD = {
    "apy": "APY",
    "apr": "APR",
    "dividend_rate": "dividend rate",
    "rate": "rate",
    "fee": "fee",
    "min_to_open": "minimum to open",
    "min_to_earn": "minimum balance to earn",
}
_ORG_WORDS = {
    "community",
    "credit",
    "union",
    "cu",
    "the",
    "bank",
    "branch",
    "branches",
    "location",
    "locations",
} | ORG_WORDS_SITE
_CHECK_NUMBER = re.compile(r"(?<![\w$.])(\d{1,2}(?:\.\d+)?)(?![\d,]|\s*(?:-?\s*(?:month|year|yr|mo)))")

_BUSINESS_W = re.compile(
    r"\b(?:business|businesses|company|companies|commercial|llc|inc|corp|corporate|corporation|employer|nonprofit|"
    r"non-profit|organization|firm|startup|partnership|my shop|my store)\b",
    re.I,
)
_PERSONAL_W = re.compile(r"\b(?:personal|individual|my|me|i|family|household|myself|our)\b", re.I)
_IRA_W = re.compile(r"\b(?:roth|traditional|rollover|sep|simple|inherited|beneficiary)\b", re.I)
_TERM_W = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half|annual|yearly|monthly|month|"
    r"months|year|years|yr|yrs|mo|mos|term|short|long)\b",
    re.I,
)
_AMOUNT_W = re.compile(r"\d|\b(?:hundred|thousand|million|grand|k)\b", re.I)


def _canonical_text(kind, attrs, family, product, segment, quals) -> str | None:
    head = product or FAMILY_NOUNS.get(family)
    if not head:
        return None
    parts = []
    if segment == "business":
        parts.append("business")
    if quals.get("ira_type"):
        parts.append(quals["ira_type"])
    parts.append(head)
    if quals.get("term_months"):
        parts.append(f"{quals['term_months']} month")
    if quals.get("purpose"):
        parts.append(quals["purpose"])
    if quals.get("model_year"):
        parts.append(f"{quals['model_year']} model year")
    if quals.get("amount"):
        parts.append(f"${quals['amount']:,.0f}")
    if quals.get("phase") == "intro":
        parts.append("introductory")
    word = next((_ATTR_WORD[a] for a in attrs if a in _ATTR_WORD), "")
    if word and not (kind == "fee" and head.lower().endswith("fee")):
        parts.append(word)
    return " ".join(parts)


def _specific_entities(ents, raw: str) -> tuple:
    low = (raw or "").lower()
    out = []
    for e in ents:
        toks = re.findall(r"[a-z0-9$]+", e.lower())
        if not toks or all(t in _ORG_WORDS for t in toks) or e.lower() not in low:
            continue
        out.append(e)
    return tuple(out[:3])


def _kind_and_attrs(q, v: dict) -> tuple[str, tuple, tuple]:
    kind = v.get("kind", q.kind)
    attrs = tuple(q.attrs)
    if "attributes" in v and v["attributes"]:
        attrs = tuple(v["attributes"])
    elif kind != q.kind:
        attrs = _KIND_ATTRS.get(kind, attrs if kind in ("rate", "contact") else ())
    excl = tuple(v.get("excluded_attributes", ()))
    if excl:
        attrs = tuple(a for a in attrs if a not in excl) or attrs
    if kind == "rate":
        attrs = tuple(a for a in attrs if a in _RATE_ATTRS) or tuple(q.attrs)
    if kind == "contact":
        attrs = tuple(a for a in attrs if a in ("phone", "routing_number"))
    if kind != "text" and not attrs:
        kind, attrs = q.kind, tuple(q.attrs)
    if kind != "text" and not attrs:
        kind = "text"
    return kind, attrs, excl


def merge(q, v: dict | None):
    """The rule-parsed `q` with the LLM's valid fields laid over it (a new Query; `q` itself is not changed)."""
    if not v:
        return q
    kind, attrs, excl = _kind_and_attrs(q, v)
    kw: dict = dict(kind=kind, attrs=attrs, intent=v.get("intent", q.intent))
    if v.get("family"):
        kw["family"] = v["family"]
    if v.get("segment"):
        kw["segment"] = v["segment"]
    quals = dict(q.qualifiers)
    quals.update(v.get("qualifiers", {}))
    kw["qualifiers"] = quals
    prod = v.get("product")
    if prod:
        kw["product"] = prod
    text = q.text
    canon = (
        _canonical_text(kind, attrs, kw.get("family", q.family), prod, kw.get("segment", q.segment), quals)
        if v.get("answerable") != "no" and kind in ("rate", "fee", "min_open", "min_earn")
        else None
    )
    if canon:
        text = canon
        kw["words"] = parse_query(canon).words
    kw["text"] = text
    if v.get("intent") == "check" and kind == "rate" and not q.numbers:
        skip = {float(quals.get("term_months", -1)), float(quals.get("term_months", -12)) / 12}
        kw["numbers"] = [float(x) for x in _CHECK_NUMBER.findall(q.raw) if float(x) not in skip]
    if v.get("as_of") and q.as_of_source != "arg":
        kw.update(as_of=v["as_of"], as_of_source="text")
    if kind == "contact":
        kw["asks_phone"] = "phone" in attrs or q.asks_phone
        kw["asks_routing"] = "routing_number" in attrs or q.asks_routing
    kw["exclude_attrs"] = excl
    kw["exclude_products"] = tuple(v.get("excluded_products", ()))
    kw["answerable"] = v.get("answerable")
    kw["reason"] = v.get("reason") if v.get("reason") != "none" else None
    kw["answer_type"] = v.get("answer_type")
    kw["entities"] = _specific_entities(v.get("entities", ()), q.raw)
    return dataclasses.replace(q, **kw)


def _word_backed(low: str, base_value, value, pattern) -> bool:
    return value == base_value or bool(pattern.search(low))


def _product_backed(prod: str, low: str, cat: dict) -> bool:
    qtok = {w[:5] for w in re.findall(r"[a-z0-9]+", low) if len(w) >= 3}
    names = {prod.lower()} | {a for a, full in (cat.get("aliases") or {}).items() if full == prod.lower()}
    ptok = {w[:5] for n in names for w in re.findall(r"[a-z0-9]+", n) if len(w) >= 3 and w not in SHORT_NAME_DROP}
    return bool(qtok & ptok)


def supported_fields(v: dict, base, raw: str, cat: dict) -> dict:
    """`v` without the LLM fields that differ from the rule parse `base` and are not backed by the question's own."""
    out = dict(v)
    low = (raw or "").lower()
    if out.get("segment") and out["segment"] != base.segment:
        ok = _BUSINESS_W.search(low) if out["segment"] == "business" else _PERSONAL_W.search(low)
        if not ok:
            out.pop("segment")
    qs = dict(out.get("qualifiers") or {})
    for key, pattern in (("ira_type", _IRA_W), ("term_months", _TERM_W), ("amount", _AMOUNT_W)):
        if key in qs and not _word_backed(low, base.qualifiers.get(key), qs[key], pattern):
            qs.pop(key)
    if "qualifiers" in out:
        out["qualifiers"] = qs
    fam = out.get("family")
    if fam and fam != base.family:
        rx = FAMILY_HINTS.get(fam) or "|".join(w[:4] for w in fam.split("_"))
        if not re.search(rx, low):
            out.pop("family")
    prod = out.get("product")
    if prod and prod.lower() != (getattr(base, "product", None) or "").lower() and not _product_backed(prod, low, cat):
        out.pop("product")
    return out


# ---- parser
PROMPT_VERSION = "p2b"
LLM_TIMEOUT = 2.8
MAX_TOKENS = 400

INTENTS = ("value", "list", "check", "change", "url", "howto", "branch", "info")
KINDS = ("rate", "fee", "min_open", "min_earn", "contact", "text")
ATTRS = ("apy", "apr", "dividend_rate", "rate", "fee", "min_to_open", "min_to_earn", "phone", "routing_number")
SEGMENTS = ("personal", "business")
PURPOSES = ("purchase", "refinance")
IRA_TYPES = ("traditional", "roth")
PHASES = ("intro", "after_intro", "intro_6mo")
ANSWER_TYPES = (
    "hours",
    "address",
    "phone",
    "amount",
    "count",
    "date",
    "list",
    "procedure",
    "yes_no",
    "explanation",
    "other",
)
ANSWERABLE = ("yes", "no", "uncertain")
REASONS = ("advice", "off_topic", "personal_account_data", "not_published_type", "nonexistent_product", "none")
HARD_NO = ("advice", "off_topic", "personal_account_data")
ENTITY_MAX_CHARS = 60
ENTITY_MAX_COUNT = 8
EMPTY_CATALOG = {"families": [], "products": {}, "branches": []}


# ---------------------------------------------------------------------------------------------------- prompt
def build_system(cat: dict) -> str:
    by_fam: dict[str, list[str]] = {}
    back = {full: short for short, full in cat.get("aliases", {}).items()}
    for fam, name in sorted(cat["products"].values()):
        by_fam.setdefault(fam, []).append(back.get(name.lower(), name.lower()))
    lines = ["Families: " + ",".join(cat["families"])]
    lines += [f"{fam}: " + "|".join(v) for fam, v in sorted(by_fam.items())]
    lines.append("Branches: " + "|".join(cat["branches"]))
    return PARSER_SYSTEM + PARSER_OMIT + "\n" + "\n".join(lines)


def build_user(question: str, latest: str) -> str:
    q = (question or "").replace("</question>", "< /question>").replace("<question>", "< question>")
    return f"<latest_snapshot_date>{latest}</latest_snapshot_date>\n<question>\n{q}\n</question>"


def build_tool(cat: dict) -> dict:
    return {
        "name": "record_query",
        "description": "Record the query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": list(INTENTS)},
                "kind": {"type": "string", "enum": list(KINDS)},
                "attributes": {"type": "array", "items": {"type": "string", "enum": list(ATTRS)}},
                "excluded_attributes": {"type": "array", "items": {"type": "string", "enum": list(ATTRS)}},
                "family": {"type": "string"},  # validated against the catalog afterwards
                "product": {"type": ["string", "null"]},
                "excluded_products": {"type": "array", "items": {"type": "string"}},
                "segment": {"type": ["string", "null"], "enum": [*SEGMENTS, None]},
                "qualifiers": {
                    "type": "object",
                    "properties": {
                        "term_months": {"type": "integer"},
                        "amount": {"type": "number"},
                        "model_year": {"type": "integer"},
                        "purpose": {"type": "string", "enum": list(PURPOSES)},
                        "ira_type": {"type": "string", "enum": list(IRA_TYPES)},
                        "phase": {"type": "string", "enum": list(PHASES)},
                    },
                },
                "as_of": {"type": ["string", "null"]},
                "entities": {"type": "array", "items": {"type": "string"}},
                "answer_type": {"type": "string", "enum": list(ANSWER_TYPES)},
                "answerable": {"type": "string", "enum": list(ANSWERABLE)},
                "reason": {"type": "string", "enum": list(REASONS)},
            },
            "required": ["intent", "kind", "answerable"],
        },
    }


# ---------------------------------------------------------------------------------------------------- validation
_JUNK = {"", "none", "null", "n/a", "na", "nil", "unknown", "undefined", "\"\"", "''", "[]", "{}"}
# (field, lowest, highest, integer) of the numeric qualifiers
_NUMERIC_QUALIFIERS = (("term_months", 1, 600, True), ("amount", 0.01, 1e10, False), ("model_year", 1990, 2100, True))
_ENUM_QUALIFIERS = (("purpose", PURPOSES), ("ira_type", IRA_TYPES), ("phase", PHASES))
_ENUM_FIELDS_HEAD = (("intent", INTENTS), ("kind", KINDS))
_ENUM_FIELDS_TAIL = (("answer_type", ANSWER_TYPES), ("answerable", ANSWERABLE), ("reason", REASONS))


def _junk(x) -> bool:
    return isinstance(x, str) and x.strip().strip("\"'`").strip().lower() in _JUNK


def _iso(s) -> str | None:
    if isinstance(s, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.strip()):
        try:
            return date.fromisoformat(s.strip()).isoformat()
        except ValueError:
            return None
    return None


def _num(x, lo, hi, integer):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if integer and (isinstance(x, float) and x != int(x)):
        return None
    return (int(x) if integer else x) if lo <= x <= hi else None


def _product(x, cat) -> str | None:
    if isinstance(x, str) and not _junk(x):
        key = x.strip().lower()
        hit = cat["products"].get(key) or cat["products"].get(cat.get("aliases", {}).get(key, ""))
        return hit[1] if hit else None
    return None


def _qualifiers(qs: dict) -> dict:
    out = {}
    for key, lo, hi, integer in _NUMERIC_QUALIFIERS:
        n = _num(qs.get(key), lo, hi, integer)
        if n is not None:
            out[key] = n
    for key, allowed in _ENUM_QUALIFIERS:
        if qs.get(key) in allowed:
            out[key] = qs[key]
    return out


def _copy_enums(obj: dict, fields, out: dict) -> None:
    for key, allowed in fields:
        if obj.get(key) in allowed:
            out[key] = obj[key]


def validate(obj, cat: dict) -> dict | None:
    """A cleaned dict holding only valid fields (None if `obj` is not a dict)."""
    if not isinstance(obj, dict):
        return None
    out: dict = {}
    _copy_enums(obj, _ENUM_FIELDS_HEAD, out)
    for key in ("attributes", "excluded_attributes"):
        v = obj.get(key)
        if isinstance(v, list):
            vals = [a for a in v if a in ATTRS]
            if vals or not v:
                out[key] = list(dict.fromkeys(vals))
    if isinstance(obj.get("family"), str) and not _junk(obj["family"]) and obj["family"] in cat["families"]:
        out["family"] = obj["family"]
    p = _product(obj.get("product"), cat)
    if p:
        out["product"] = p
    if isinstance(obj.get("excluded_products"), list):
        ex = [_product(x, cat) for x in obj["excluded_products"]]
        out["excluded_products"] = [x for x in dict.fromkeys(ex) if x]
    _copy_enums(obj, (("segment", SEGMENTS),), out)
    if isinstance(obj.get("qualifiers"), dict):
        out["qualifiers"] = _qualifiers(obj["qualifiers"])
    d = _iso(obj.get("as_of"))
    if d:
        out["as_of"] = d
    if isinstance(obj.get("entities"), list):
        out["entities"] = [
            e.strip()
            for e in obj["entities"]
            if isinstance(e, str) and not _junk(e) and len(e.strip()) <= ENTITY_MAX_CHARS
        ][:ENTITY_MAX_COUNT]
    _copy_enums(obj, _ENUM_FIELDS_TAIL, out)
    return out


# ---------------------------------------------------------------------------------------------------- the parser
class LLMParser(Session):
    """One per knowledge base: the prompt, tool, cache key and validation of a question against the catalog."""

    version = PROMPT_VERSION
    max_tokens = MAX_TOKENS

    def __init__(self, catalog: dict, latest: str, settings: LLMSettings, timeout: float | None = None):
        super().__init__(settings, float(timeout if timeout is not None else LLM_TIMEOUT))
        self.cat = catalog
        self.latest = latest
        self._digest = catalog_digest(catalog)

    @cached_property
    def system(self) -> str:
        return build_system(self.cat)

    @cached_property
    def tool(self) -> dict:
        return build_tool(self.cat)

    def cache_key(self, text: str, as_of, model: str) -> str:
        return hashlib.sha256(
            json.dumps([self.version, model, self._digest, normalize_question(text), str(as_of or "")]).encode()
        ).hexdigest()[:32]

    def user(self, text: str, payload) -> str:
        return build_user(text, self.latest)

    def validate(self, raw):
        return validate(raw, self.cat)


def make_parser(kb, folder=None, timeout: float | None = None, **settings) -> LLMParser:
    """The parser for a knowledge base; inactive (and free."""
    resolved = LLMSettings.resolve(folder=folder or SOLUTION_DIR, **settings)
    if not resolved.active:
        return LLMParser(EMPTY_CATALOG, "", resolved)
    return LLMParser(build_catalog(kb), kb.views[-1].taken_on if kb.views else "", resolved, timeout)


__all__ = ["HARD_NO", "Budget", "LLMParser", "build_catalog", "make_parser", "merge", "supported_fields", "validate"]
