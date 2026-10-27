"""Facts from fee-schedule PDF lines and from 'label: value' lines and one-rate sentences."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.dates import effective_dates, first_date
from ..core.textutil import SENT_BREAK_OR_LINE, strip_md_links
from ..site_profile import is_business_url
from .facts import WEAK_TOKENS, Conf, Fact, ProductRegistry, make_key
from .normalize import (
    LOAN_FAMILIES,
    canon_name,
    extract_qualifiers,
    family_of,
    finish_product,
    parse_value,
    strip_segment,
    tokens,
)

# ---- fee_lines
_FEE_LINE = re.compile(
    r"^(?P<label>[^$|]{3,140}?)\s*\|?\s*(?P<amt>\$\s?\d[\d,]*(?:\.\d\d)?)\s*\*?\s*(?P<unit>(?:per|each)\b[^.$|]"
    r"{0,40}?)?\s*(?:\([^)]*\))?\s*(?P<tail>(?:for|to|from|with|on|if|when|after|before)\b[^$|]{0,40})?\s*$",
    re.I,
)
_SMALL_WORDS = {"and", "of", "or", "the", "for", "to", "a", "an", "in", "on", "with", "&", "-"}
MAX_HEADING_WORDS = 9
MAX_HEADING_CHARS = 70
EFFECTIVE_SCAN_CHARS = 600


def _is_heading(line: str) -> bool:
    if "|" in line or len(line) >= MAX_HEADING_CHARS or line.endswith((".", ":", ",", ";", ")")) or "•" in line:
        return False
    words = line.split()
    return 0 < len(words) <= MAX_HEADING_WORDS and all(
        w.lower() in _SMALL_WORDS or w[:1].isupper() or w[:1].isdigit() for w in words
    )


def _label(m: re.Match, lines: list[str], i: int) -> str:
    label = m.group("label").strip(" -:*")
    if m.group("tail"):
        label = f"{label} {m.group('tail').strip(' -:*')}"  # 'Notary services $10.00 for non-members'
    if label[:1] in "(" or label[:1].islower():
        prev = lines[i - 1].strip() if i else ""
        if prev and "$" not in prev:
            label = f"{prev} {label}"
    return label


def extract_fee_lines(page, stats) -> list[Fact]:
    """Observations from the fee lines of a fee-schedule PDF."""
    out: list[Fact] = []
    lines = page.parsed.text.split("\n")
    section = ""
    eff = (effective_dates(page.parsed.text[:EFFECTIVE_SCAN_CHARS]) or [None])[0]
    flattened = page.parsed.junk_table
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if "$" not in line:
            if _is_heading(line):
                section = line
            continue
        m = _FEE_LINE.match(line)
        if not m:
            stats["fee_lines_unparsed"] += 1
            continue
        label = _label(m, lines, i)
        amt = parse_value(m.group("amt"))
        if not label or amt is None:
            continue
        unit = (m.group("unit") or "").strip().lower()
        q = {"unit": re.sub(r"\W+", "_", unit)} if unit else {}
        name, biz = strip_segment(label)
        if biz or "business" in section.lower():
            q["segment"] = "business"
        prod = canon_name(re.sub(r"\([^)]*\)", " ", name)) or canon_name(name)
        out.append(
            Fact.on_page(
                page,
                key=make_key(prod, "fee", q),
                product_raw=label,
                family="fee",
                value=amt.lo,
                unit="$",
                raw_value=line,
                heading=section,
                caption=section,
                column="fee",
                row_text=line,
                effective_date=eff,
                effective_scope="chunk" if eff else None,
                source_kind="fee_line",
                confidence=Conf.FEE_LINE_FLATTENED if flattened else Conf.FEE_LINE,
                flags=("pdf_flattened",) if flattened else (),
            )
        )
    stats["fee_lines"] += len(out)
    return out


# ---- line_facts
_ATTR = r"APY|APR|annual percentage yield|annual percentage rate|dividend rate|interest rate|dividend|rate"
_ATTR_NAME = {
    "apy": "apy",
    "annual percentage yield": "apy",
    "apr": "apr",
    "annual percentage rate": "apr",
    "dividend rate": "dividend_rate",
    "dividend": "dividend_rate",
    "interest rate": "rate",
    "rate": "rate",
}
_NUM = r"\d+(?:\.\d+)?"
# attribute word before the value: '6-month certificate APY: 2.95%', 'APY of 3.5%'
_FORM_A = re.compile(
    rf"(?P<label>[^%$]{{0,110}}?)\b(?P<attr>{_ATTR})(?![A-Za-z])\*?\d?\s*(?:[:=]|of|is|at|-|–)?\s*(?P<v>{_NUM})\s*%",
    re.I,
)
# value before the attribute word: 'earn 4.10% APY on our 17-month certificate special'
_FORM_B = re.compile(rf"(?P<v>{_NUM})\s*%\s*(?:\*+|\[\*\]|\d{{1,2}}(?![\d%]))?\s*(?P<attr>{_ATTR})(?![A-Za-z])", re.I)
# 'label: value' (value first, attribute in the label or right after the value)
_FORM_L = re.compile(rf"^(?P<label>[^:%$]{{3,110}}?)\s*:\s*(?P<v>{_NUM}\s*%|\$\s?\d[\d,]*(?:\.\d\d)?)(?P<tail>.*)$")
_FEE_WORD = re.compile(r"\b(fees?|charges?)\b", re.I)
_EXAMPLE = re.compile(
    r"\b(for example|e\.g\.|for instance|illustrat\w+|hypothetical|sample|suppose|assum\w+|payment example|"
    r"if you (?:borrow|deposit|invest|save|put)|calculator|estimate\w*|average|return of)\b",
    re.I,
)
_HIST = re.compile(
    r"\b(was|were|had been|previously|formerly|used to|no longer|last (?:year|month|quarter)|back in|ago|once)\b", re.I
)
_PAST_YEAR = re.compile(r"\b(?:in|during|since|throughout|of|for)\s+(?:the\s+)?(?:year\s+)?(19\d\d|20\d\d)\b", re.I)
_OFFER_TRIGGER = re.compile(
    r"\b(ended|expired|expires?|ends?|ending|valid (?:through|until|thru)|good through|through|until|thru)\b[\s:]"
    r"*(?:on\s+)?",
    re.I,
)
_PROMO = re.compile(
    r"\b(special|promo\w*|limited[- ]time|offer(?:s|ed)?|bonus|introductory|intro|new[- ]member|summer|spring|winter|"
    r"fall|holiday|sale|deal|exclusive|first (?:\d+|six|three|twelve|nine) months?)\b",
    re.I,
)
_FLOOR = re.compile(r"\b(as low as|starting at|up to|as high as|from|as much as)\s*$", re.I)
_FILLER = re.compile(
    r"\b(earn(?:s|ed|ing)?|get|gets|receive|enjoy|with|our|your|the|an?|on|for|at|is|are|offers?|offering|offered|"
    r"new|great|current|now|pays?|paying|has|have|features?|provides?|that|this|these|those|just|only|"
    r"as low as|starting at|up to|as high as|from|apy|apr|rate|rates|of|in|to|and|or|you|can|will|"
    r"could|when|"
    r"open|opens|opened|available|today|right|per|year|annual|percentage|yield)\b",
    re.I,
)
_STOP_PHRASE = re.compile(
    r"\b(when|if|that|which|and|with|for|by|as|after|until|through|while|but|at|this|today|now|all|every|each)\b|"
    r"[,;.(]",
    re.I,
)
_PER = re.compile(r"\b(per [a-z]+|each [a-z]+)\b", re.I)
_DATE_RX = re.compile(r"\b(?:20\d\d-\d\d-\d\d|\d{1,2}/\d{1,2}/\d{2,4}|[A-Z][a-z]{2,8}\.? \d{1,2},? \d{4})\b")
_JUNK_FEE = {
    "fee",
    "fees",
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
    "total",
    "annual",
}
_RANGE_AFTER = re.compile(rf"\s*(?:–|—|-|to)\s*\$?{_NUM}")


MAX_LINE_CHARS = 700
MAX_SHORT_LINE_CHARS = 170
MAX_SHORT_LINE_WORDS = 14
MAX_RATE_PERCENT = 60
MAX_FEE_DOLLARS = 100000
MAX_FEE_NAME_WORDS = 6
RAW_VALUE_CHARS = 160
ROW_TEXT_CHARS = 240
CAPTION_CHARS = 120
OFFER_DATE_CHARS = 28


@dataclass
class _Reading:
    attr: str | None
    unit: str
    value: float
    label: str
    value_start: int
    fee: bool = False
    per_unit: str = ""
    alternatives: list = field(default_factory=list)  # further candidate product phrases


@dataclass
class _Product:
    name: str
    family: str
    new: bool
    quals: dict
    business: bool
    label: str


def _clean(line: str) -> str:
    s = strip_md_links(line)
    s = re.sub(r"^\s*(?:[-*•–·]|#{1,6}|>)\s+", "", s)
    s = s.replace("**", "").replace("__", "").strip(" _\t")
    return re.sub(r"\s+", " ", s)


def _offer_end(s: str) -> str | None:
    for m in _OFFER_TRIGGER.finditer(s):
        seg = s[m.end() : m.end() + OFFER_DATE_CHARS]
        d = first_date(seg)
        if d and re.match(r"\s*(?:\d|[A-Z][a-z]{2})", seg):
            return d
    return None


def _phrases(pre: str, rest: str) -> list[str]:
    out = []
    for m in re.finditer(r"\b(?:on|for|with|in|from|our|your)\s+(?:our|the|a|an|your|any|every|new)?\s*", rest, re.I):
        tail = rest[m.end() :]
        cut = _STOP_PHRASE.search(tail)
        out.append((tail[: cut.start()] if cut else tail).strip())
    clauses = [c for c in re.split(r"[,;:]", pre) if c.strip()]
    out += [
        re.sub(r"\b(earn|pay|offer|get|has|have|feature|provide)\w*\b.*$", "", c, flags=re.I).strip()
        for c in reversed(clauses)
    ]
    return [x for x in out if x]


def _resolve(
    label: str, sent: str, registry: ProductRegistry, known_fams: set, attr, fee: bool, per_unit: str
) -> _Product | None:
    label = re.sub(r"\([^)]*\)", " ", label)
    q, res = extract_qualifiers(label)
    whole_q, _ = extract_qualifiers(_DATE_RX.sub(" ", sent))
    if re.search(r"\b(?:get|earn|receive|win)\s+\$|\bbonus\b", sent, re.I):
        whole_q.pop("amount_range", None)  # a cash bonus is not a balance tier
    for k in ("term_months", "model_year", "purpose", "ira_type", "amount_range"):
        if k not in q and k in whole_q and not (fee and k == "amount_range"):
            q[k] = whole_q[k]
    name, biz = strip_segment(res)
    name = re.sub(r"\b(?:fees?|charges?)\b", " ", name, flags=re.I)
    if not fee:
        name = _FILLER.sub(" ", name)
    name = " ".join(name.split())
    if fee:
        cname = " ".join(canon_name(name).split()[:MAX_FEE_NAME_WORDS])
        if not cname or cname in _JUNK_FEE or "fee" in known_fams:
            return None
        return _Product(cname, "fee", True, {"unit": per_unit} if per_unit else {}, biz, label)
    fam = family_of(name, label)
    if not fam:
        return None
    prod = registry.match_phrase(name) or registry.best_in(tokens(name))
    new = False
    if prod is None:
        if fam in known_fams:
            return None
        cname = canon_name(name)
        if not cname or all(w in WEAK_TOKENS for w in cname.split()):
            return None
        prod, new = finish_product(cname, fam), True
    else:
        fam = registry.family(prod)
    if attr in ("apr", "apy") and (attr == "apr") != (fam in LOAN_FAMILIES):
        return None  # APR belongs to loans, APY to deposits
    kq = {k: v for k, v in q.items() if k in ("term_months", "model_year", "purpose", "ira_type")}
    if "amount_range" in q:
        kq["loan_amount_range" if fam in LOAN_FAMILIES else "balance_range"] = q["amount_range"]
    if fam not in ("certificate", "ira") and fam not in LOAN_FAMILIES or fam == "credit_card":
        kq.pop("term_months", None)
    return _Product(prod, fam, new, kq, biz, label)


# ---------------------------------------------------------------------------------------------------- reading a
# statement
def _split_tag(sent: str) -> tuple[str, str]:
    mt = re.match(r"^(?P<tag>[^:%$\d]{2,40}):\s+(?P<body>.+)$", sent)
    if mt and not _FORM_L.match(sent) and (_FORM_A.search(mt.group("body")) or _FORM_B.search(mt.group("body"))):
        return mt.group("tag"), mt.group("body")
    return "", sent


def _read_labelled(ml: re.Match, ch, page) -> _Reading | None:
    lab, v, tail = ml.group("label"), ml.group("v"), ml.group("tail")
    if v.startswith("$"):
        if not (_FEE_WORD.search(lab) or _FEE_WORD.search(ch.heading_path or "") or _FEE_WORD.search(page.title or "")):
            return None
        per = _PER.search(tail)
        return _Reading(
            None,
            "$",
            float(re.sub(r"[^\d.]", "", v)),
            lab,
            ml.start("v"),
            True,
            per.group(1).lower().replace(" ", "_") if per else "",
        )
    ma = re.search(rf"\b({_ATTR})\b", lab, re.I) or re.match(rf"\s*\*?\d?\s*({_ATTR})\b", tail, re.I)
    if not ma or _RANGE_AFTER.match(tail):
        return None
    return _Reading(_ATTR_NAME[ma.group(1).lower()], "%", float(v.rstrip("% ").strip()), lab, ml.start("v"))


def _read_rate(body: str) -> _Reading | None:
    ma, mb = _FORM_A.search(body), _FORM_B.search(body)
    if ma and (not mb or ma.start("v") <= mb.start("v")):
        if _RANGE_AFTER.match(body[ma.end() - 1 :]) or re.search(
            rf"{_NUM}\s*%\s*(?:–|—|-|to)\s*$", body[: ma.start("v")]
        ):
            return None
        return _Reading(
            _ATTR_NAME[ma.group("attr").lower()], "%", float(ma.group("v")), ma.group("label"), ma.start("v")
        )
    if mb:
        pre, rest = body[: mb.start()], body[mb.end() :]
        if re.search(r"%\s*(?:–|—|-|to)\s*$", pre) or _RANGE_AFTER.match(
            body[mb.start("v") + len(mb.group("v")) :].lstrip("%").lstrip()
        ):
            return None
        alts = _phrases(pre, rest)
        return _Reading(
            _ATTR_NAME[mb.group("attr").lower()],
            "%",
            float(mb.group("v")),
            alts[0] if alts else "",
            mb.start("v"),
            alternatives=alts,
        )
    return None


def _read_value(body: str, ch, page) -> _Reading | None:
    ml = _FORM_L.match(body)
    return _read_labelled(ml, ch, page) if ml else _read_rate(body)


def _is_historical(sent: str, year: int, offer_end: str | None) -> bool:
    if _HIST.search(sent) and not offer_end:
        return True
    py = _PAST_YEAR.search(sent)
    return bool(
        py
        and int(py.group(1)) < year
        and not re.search(r"\b(effective|offer|expires?|ends?|through|until)\b", sent, re.I)
    )


def _implausible(r: _Reading, sent: str, kind: str) -> bool:
    if kind == "sentence" and len(re.findall(rf"{_NUM}\s*%", sent)) >= 2:
        return True
    return (
        r.value <= 0 or (r.unit == "%" and r.value > MAX_RATE_PERCENT) or (r.unit == "$" and r.value > MAX_FEE_DOLLARS)
    )


def _trust(
    kind: str, ptype: str, flags: list, new: bool, promo: bool, floor: bool, offer_end: str | None, taken_on: str
) -> float:
    conf = Conf.LINE if kind == "line" else Conf.SENTENCE
    flags.append("line" if kind == "line" else "sentence")
    for applies, flag in ((not new, "known_product"), (promo, "promo"), (floor, "from_rate")):
        if applies:
            conf = min(conf, Conf.SUSPECT)
            flags.append(flag)
    if ptype in ("blog", "newsroom"):
        conf = min(conf, Conf.STALE_CEILING)
    if offer_end:
        flags.append("offer_window")
        if offer_end < (taken_on or "9999")[:10]:
            conf = min(conf, Conf.STALE_CEILING)
            flags.append("expired_offer")
    return conf


def _statement(
    page, ch, sent: str, registry: ProductRegistry, known_fams: set, year: int, on_biz: bool, pub, stats
) -> Fact | None:
    if _EXAMPLE.search(sent):
        return None
    tag, body = _split_tag(sent)
    kind = (
        "line"
        if len(sent) <= MAX_SHORT_LINE_CHARS
        and (tag or _FORM_L.match(sent) or len(sent.split()) <= MAX_SHORT_LINE_WORDS)
        else "sentence"
    )
    reading = _read_value(body, ch, page)
    if reading is None or _implausible(reading, sent, kind):
        return None
    offer_end = _offer_end(sent)
    if _is_historical(sent, year, offer_end):
        stats["line_skipped_historical"] += 1
        return None
    floor = bool(_FLOOR.search(body[: reading.value_start]))
    label = reading.label
    promo = bool(_PROMO.search(tag)) or (
        not reading.fee and bool(_PROMO.search(body.replace(label.strip(), " ") if label.strip() else body))
    )
    candidates = [label] + reading.alternatives[1:] + ([tag] if tag and not reading.fee else [])
    resolved = (_resolve(c, sent, registry, known_fams, reading.attr, reading.fee, reading.per_unit) for c in candidates)
    product = next((p for p in resolved if p is not None), None)
    if product is None:
        return None
    quals = product.quals
    if product.business or on_biz:
        quals["segment"] = "business"
    flags: list[str] = []
    conf = _trust(kind, page.page_type, flags, product.new, promo, floor, offer_end, page.taken_on)
    return Fact.in_chunk(
        page, ch, key=make_key(product.name, "fee" if reading.fee else reading.attr, quals),
        product_raw=product.label.strip() or product.name, family=product.family, value=reading.value,
        unit=reading.unit, raw_value=sent[:RAW_VALUE_CHARS], caption=(tag or product.label)[:CAPTION_CHARS],
        row_text=sent[:ROW_TEXT_CHARS], effective_date=pub, effective_scope="chunk" if pub else None,
        source_kind=kind, confidence=round(conf, 2), flags=tuple(flags), offer_end=offer_end)


def extract_lines(page, registry: ProductRegistry, known_fams: set, stats) -> list[Fact]:
    """Line / sentence observations of one page version (see module docstring)."""
    out: list[Fact] = []
    ptype = page.page_type
    if page.parsed.fmt == "pdf" or ptype == "form_legal" or page.parsed.listing:
        return out
    year = int((page.taken_on or "2000")[:4])
    on_biz = is_business_url(page.url)
    pub = page.publish_date if ptype in ("blog", "newsroom") else None
    for ch in page.parsed.chunks:
        for raw in ch.text.split("\n"):
            if not raw.strip() or raw.lstrip().startswith("|") or ("%" not in raw and "$" not in raw):
                continue
            line = _clean(raw)
            if len(line) > MAX_LINE_CHARS:
                continue
            if _EXAMPLE.search(line):
                stats["line_skipped_example"] += 1
                continue
            for sent in (
                [line]
                if len(line) <= MAX_SHORT_LINE_CHARS
                else [x.strip() for x in SENT_BREAK_OR_LINE.split(line) if x.strip()]
            ):
                f = _statement(page, ch, sent, registry, known_fams, year, on_biz, pub, stats)
                if f is not None:
                    out.append(f)
    seen, res = set(), []
    for f in out:
        sig = (f.key, f.value)
        if sig not in seen:
            seen.add(sig)
            res.append(f)
    stats["line_facts"] += len(res)
    return res
