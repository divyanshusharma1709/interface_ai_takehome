"""Conservative prose facts: 'x% APR/APY' tied to a known table product, 'fee' amounts, the routing number, phone
numbers on contact pages."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from ..core.textutil import SENT_BREAK_OR_LINE
from ..site_profile import is_business_url
from .facts import Conf, Fact, ProductRegistry, make_key
from .normalize import LOAN_FAMILIES, canon_name, extract_qualifiers, tokens

_PROSE_PCT = re.compile(
    r"(\d+(?:\.\d+)?)\s*%(?:\s*(?:–|—|-|to)\s*(\d+(?:\.\d+)?)\s*%)?\s*(?:\*|\d)?\s*(APR|APY|"
    r"annual percentage (?:rate|yield))",
    re.I,
)
_ROUTING = re.compile(r"routing(?:/transit)?\s*(?:number|#|no\.?)?[^0-9\n]{0,40}?(\d{9})\b", re.I)
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[-. ])?\(?(\d{3})\)?[-. ](\d{3})[-. ](\d{4})(?!\d)(?:\s*(?:ext\.?|x|extension)\s*(\d{1,5}))?", re.I
)
_FEE_A = re.compile(
    r"\$\s?(\d[\d,]*(?:\.\d\d)?)\*?\s+((?:[A-Za-z][A-Za-z-]*\s+){0,3}?)fee\b(?:\*)?(?:\s+(per [a-z]+|each))?", re.I
)
_FEE_B = re.compile(
    r"\b((?:[A-Za-z][A-Za-z-]*\s+){1,3}?)fee\s*(?:is|of|:)\s*(?:only\s+)?\$\s?(\d[\d,]*(?:\.\d\d)?)"
    r"\b(?:\s+(per [a-z]+|each))?",
    re.I,
)
_FEE_PCT = re.compile(
    r"\b((?:[A-Za-z][A-Za-z-]*\s+){1,3}?)fee\s*[-—–:]?\s*(\d+(?:\.\d+)?)\s*%(?:\s+of\s+(?:the\s+|each\s+|every\s+)"
    r"?([a-z]+))?",
    re.I,
)
_JUNK_FEE_LABELS = {
    "fee",
    "monthly",
    "service",
    "other",
    "late",
    "yearly",
    "additional",
    "extra",
    "applicable",
    "any",
    "this",
}
_NEWS_TYPES = ("blog", "newsroom")
CAPTION_CHARS = 120
LABEL_WORDS_BEFORE_PHONE = 4
LABEL_CHARS_BEFORE_PHONE = 45
FEE_LABEL_WORDS = 3
PROMO_WINDOW_CHARS = 30


def _contact_page(page) -> bool:
    return page.page_type == "branch_contact" or urlparse(page.url).path.rstrip("/") in ("", "/contact")


def _routing_facts(text: str, make) -> list[Fact]:
    return [
        make(
            key=make_key("credit union", "routing_number", {}),
            product_raw="routing number",
            family="contact",
            value=None,
            unit="",
            raw_value=m.group(0)[:80],
            value_text=m.group(1),
            confidence=Conf.ROUTING_NUMBER,
        )
        for m in _ROUTING.finditer(text)
    ]


def _phone_facts(text: str, make) -> list[Fact]:
    out = []
    for m in _PHONE.finditer(text):
        pre = re.sub(r"[^A-Za-z ]+", " ", text[max(0, m.start() - LABEL_CHARS_BEFORE_PHONE) : m.start()]).split()[
            -LABEL_WORDS_BEFORE_PHONE:
        ]
        q = {"label": " ".join(w.lower() for w in pre)}
        if m.group(4):
            q["ext"] = m.group(4)
        out.append(
            make(
                key=make_key("credit union", "phone", q),
                product_raw="phone",
                family="contact",
                value=None,
                unit="",
                raw_value=m.group(0),
                value_text=f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
                confidence=Conf.PHONE,
            )
        )
    return out


def _rate_facts(sent: str, heading_path: str, page, registry: ProductRegistry, make) -> list[Fact]:
    out = []
    ctx = tokens(sent)
    last_head = heading_path.split(" > ")[-1] if heading_path else ""
    prod = None
    for m in _PROSE_PCT.finditer(sent):
        if prod is None:
            prod = registry.best_in(ctx)
        if prod is None:
            hp = registry.best_in(tokens(last_head))
            prod = hp if hp and registry.family(hp) in LOAN_FAMILIES else None
        if prod is None:
            break
        attr = "apr" if m.group(3).lower().startswith(("apr", "annual percentage rate")) else "apy"
        family = registry.family(prod)
        if (attr == "apr") != (family in LOAN_FAMILIES):  # APR belongs to loans, APY to deposits
            continue
        q, _ = extract_qualifiers(sent)
        kq = {k: v for k, v in q.items() if k in ("term_months", "model_year", "purpose")}
        if family == "credit_card":
            kq.pop("term_months", None)
        near = sent[max(0, m.start() - PROMO_WINDOW_CHARS) : m.end() + PROMO_WINDOW_CHARS]
        if re.search(r"\bintro", near, re.I) and "intro" not in prod and not re.search(r"after that", near, re.I):
            kq["phase"] = "intro"
        if is_business_url(page.url):
            kq["segment"] = "business"
        out.append(
            make(
                key=make_key(prod, attr, kq),
                product_raw=prod,
                family=family,
                value=float(m.group(1)),
                value_hi=float(m.group(2)) if m.group(2) else None,
                unit="%",
                raw_value=m.group(0),
                caption=sent[:CAPTION_CHARS],
                confidence=Conf.PROSE_RATE,
                flags=("prose",),
            )
        )
    return out


def _fee_label(words: str) -> str:
    label = canon_name(words)
    label = re.sub(r"\b(a|an|the|your|no|free|that|with|and)\b", " ", label).strip()
    return " ".join(label.split()[-FEE_LABEL_WORDS:])


def _fee_facts(sent: str, make) -> list[Fact]:
    out = []
    for rx, amount_first in ((_FEE_A, True), (_FEE_B, False)):
        for m in rx.finditer(sent):
            amt, lab, per = (
                (m.group(1), m.group(2), m.group(3)) if amount_first else (m.group(2), m.group(1), m.group(3))
            )
            label = _fee_label(lab)
            if not label or label in _JUNK_FEE_LABELS:
                continue
            q = {"unit": per.lower().replace(" ", "_")} if per else {}
            out.append(
                make(
                    key=make_key(label, "fee", q),
                    product_raw=m.group(0),
                    family="fee",
                    value=float(amt.replace(",", "")),
                    unit="$",
                    raw_value=m.group(0),
                    caption=sent[:CAPTION_CHARS],
                    confidence=Conf.PROSE_FEE,
                    flags=("prose",),
                )
            )
    return out


def _percent_fee_facts(sent: str, make) -> list[Fact]:
    out = []
    for m in _FEE_PCT.finditer(sent):
        label = " ".join(canon_name(m.group(1)).split()[-FEE_LABEL_WORDS:])
        if not label or label in _JUNK_FEE_LABELS:
            continue
        per = re.sub(r"\W+", "_", ("of " + m.group(3)).strip()) if m.group(3) else ""
        out.append(
            make(
                key=make_key(label, "fee", {"unit": per} if per else {}),
                product_raw=m.group(0),
                family="fee",
                value=float(m.group(2)),
                unit="%",
                raw_value=m.group(0),
                caption=sent[:CAPTION_CHARS],
                confidence=Conf.PROSE_PERCENT_FEE,
                flags=("prose",),
            )
        )
    return out


def _sentence_facts(sent: str, ch, page, registry: ProductRegistry, make) -> list[Fact]:
    out = []
    if "%" in sent and registry:
        out += _rate_facts(sent, ch.heading_path, page, registry, make)
    if page.page_type not in _NEWS_TYPES and "fee" in sent.lower():
        if "$" in sent:
            out += _fee_facts(sent, make)
        if "%" in sent:
            out += _percent_fee_facts(sent, make)
    return out


def extract_prose(page, registry: ProductRegistry, stats) -> list[Fact]:
    """Conservative prose observations of one page version."""
    if page.page_type == "form_legal" and page.parsed.fmt == "pdf":
        return []
    out: list[Fact] = []
    contact = _contact_page(page)
    for ch in page.parsed.chunks:
        text = "\n".join(l for l in ch.text.split("\n") if not l.startswith("|"))
        if not text.strip():
            continue
        make = lambda **fields: Fact.in_chunk(page, ch, source_kind="prose", **fields)
        out += _routing_facts(text, make)
        if contact:
            out += _phone_facts(text, make)
        for sent in (s.strip() for s in SENT_BREAK_OR_LINE.split(text)):
            if sent:
                out += _sentence_facts(sent, ch, page, registry, make)
    seen, res = set(), []
    for f in out:
        sig = (f.key, f.value, f.value_hi, f.value_text)
        if sig not in seen:
            seen.add(sig)
            res.append(f)
    stats["prose_facts"] += len(res)
    return res
