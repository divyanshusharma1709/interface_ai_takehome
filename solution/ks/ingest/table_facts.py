"""Facts from the Markdown tables of a page: table parsing, column roles, one fact per value cell."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.dates import effective_dates
from ..core.textutil import drop_final_s, strip_md_links
from ..site_profile import is_business_url
from .chunks import Chunk
from .facts import FAMILY_TOKENS, WEAK_TOKENS, Conf, Fact, make_key
from .normalize import (
    LOAN_FAMILIES,
    canon_name,
    extract_qualifiers,
    family_of,
    finish_product,
    has_family_keyword,
    parse_value,
    strip_segment,
)

# ---- tables
_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_HDR_FOOTNOTE = re.compile(r"(?<=[A-Za-z)%])\s?([0-9*†‡¹²³])\s*$")
_FN_LINE = re.compile(r"^[_\s]*([0-9])\s*[_\s]*([A-Z][^:.\n]{2,80}?)\s*:")
_HEADER_WORDS = re.compile(
    r"\b(rate|apy|apr|yield|term|tier|balance|min|max|minimum|dividend|amount|program|type|"
    r"product|certificate|category|loan|account|points|payment|fee|month|year)\b",
    re.I,
)


def plain_cell(s: str) -> str:
    """Cell text without links, <br>, markdown escapes and surplus whitespace."""
    s = strip_md_links(s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = s.replace("\\*", "").replace("*", "").replace("\\", "")
    return " ".join(s.split())


def split_row(line: str) -> list[str]:
    r"""Split a Markdown table line into cells (outer pipes dropped; `\|` kept inside a cell)."""
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    return [plain_cell(c) for c in re.split(r"(?<!\\)\|", s)]


def header_footnote(h: str) -> tuple[str, str]:
    """('Annual Percentage Yield (APY)', '1') for 'Annual Percentage Yield (APY)1'; ('..', '') if none."""
    m = _HDR_FOOTNOTE.search(h)
    if m and re.search(r"[A-Za-z)]", h[: m.start()]):
        return h[: m.start()].strip(), m.group(1)
    return h.strip(), ""


@dataclass
class Table:
    """One Markdown table plus the text around it (all inside one chunk, or in the chunks just before)."""

    headers: list[str]  # footnote markers stripped
    header_fn: list[str]  # footnote marker per header ('' if none)
    rows: list[list[str]]
    row_groups: list[str]
    before: list[str]
    after: list[str]
    prev_lines: list[str]
    heading_path: list[str]
    n_skipped_rows: int = 0  # repeated header rows etc. dropped

    @property
    def footnotes(self) -> dict[str, str]:
        """Footnote number -> label for 'N<Label>:' lines after the table (e.g."""
        out: dict[str, str] = {}
        for l in self.after:
            m = _FN_LINE.match(l)
            if m:
                out.setdefault(m.group(1), m.group(2).strip())
        return out


def _is_header_like(cells: list[str]) -> bool:
    filled = [c for c in cells if c]
    return (
        len(filled) >= 2
        and not any(re.search(r"\d|\$|%", c) for c in filled)
        and all(_HEADER_WORDS.search(c) for c in filled)
    )


def short_label_lines(lines: list[str], limit: int = 12) -> list[str]:
    """Short caption-like lines (no sentence punctuation, links or bullets) from the tail of `lines`."""
    out = []
    for l in lines[-60:]:
        s = l.strip().strip("_").strip()
        if (
            s
            and len(s) <= 90
            and not s.startswith(("|", "-", "#", "[", ">", "!"))
            and not s.endswith((".", ":", "?"))
            and "](" not in s
            and len(s.split()) <= 12
            and not re.search(r"[$%]", s)
        ):
            out.append(plain_cell(s))
    return out[-limit:]


def _table_spans(lines: list[str]) -> list[tuple[int, int]]:
    """(first, end) line numbers of each run of '|' lines."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _read_block(block: list[str]) -> tuple[list, list, list, list, int]:
    """(headers, header footnotes, rows, section label of each row, skipped lines) of one table block."""
    hdr_cells = split_row(block[0])
    body_start = 2 if len(block) > 1 and _SEP.match(block[1]) else 1
    headers, fns = [], []
    for h in hdr_cells:
        base, fn = header_footnote(h)
        headers.append(base)
        fns.append(fn)
    rows, groups, skipped = [], [], 0
    group = ""
    for line in block[body_start:]:
        if _SEP.match(line):
            continue
        cells = split_row(line)
        cells += [""] * (len(headers) - len(cells))
        norm = [header_footnote(c)[0].lower() for c in cells]
        if norm == [h.lower() for h in headers] or (_is_header_like(cells) and not rows):
            if norm != [h.lower() for h in headers]:
                headers = [f"{h} {header_footnote(c)[0]}".strip() for h, c in zip(headers, cells + [""] * len(headers))]
            skipped += 1
            continue
        filled = [c for c in cells if c]
        if len(filled) == 1 and len(cells) > 1 and not re.search(r"[%$]\s*\d|\d\s*%", filled[0]) \
                and not re.search(r"\d", filled[0]):
            group = filled[0]
            skipped += 1
            continue
        rows.append(cells)
        groups.append(group)
    return headers, fns, rows, groups, skipped


def parse_tables(chunk: Chunk, prev_chunks: list[Chunk] | None = None) -> list[Table]:
    """All Markdown tables in a chunk, with surrounding lines and the page's preceding chunk tails."""
    lines = chunk.text.split("\n")
    prev_lines = [l for pc in (prev_chunks or [])[-3:] for l in pc.text.split("\n") if l.strip()]
    spans = _table_spans(lines)
    tables: list[Table] = []
    for k, (a, b) in enumerate(spans):
        if b - a < 2:
            continue
        headers, fns, rows, groups, skipped = _read_block(lines[a:b])
        before = [l for l in lines[spans[k - 1][1] if k else 0:a] if l.strip()]
        after = [l for l in lines[b:spans[k + 1][0] if k + 1 < len(spans) else len(lines)] if l.strip()]
        tables.append(Table(headers, fns, rows, groups, before, after, prev_lines, chunk.heading_path.split(" > "),
                            skipped))
    return tables


# ---- table_facts
# (header pattern, attribute, numeric hint)
_COLS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"payment example"), "payment_example", "$"),
    (re.compile(r"\bpoints?\b"), "points", "n"),
    (re.compile(r"min\w*\.? *(balance )?to open|minimum (deposit|opening)|opening deposit"), "min_to_open", "$"),
    (re.compile(r"min\w*\.? *(balance )?to earn"), "min_to_earn", "$"),
    (re.compile(r"\bmin\w*\.? balance"), "min_balance", "$"),
    (re.compile(r"\bmax\w*\.? balance"), "max_balance", "$"),
    (re.compile(r"dividend"), "dividend_rate", "%"),
    (re.compile(r"\bapy\b|annual percentage yield"), "apy", "%"),
    (re.compile(r"\bapr\b|annual percentage rate"), "apr", "%"),
    (re.compile(r"^(interest )?rate\b|\binterest rate\b"), "rate", "%"),
    (re.compile(r"\bfee\b"), "fee", "$"),
]
_PHASE_INTRO = re.compile(r"(?:1st|first) (\d+) months?|introductory|promotional rate", re.I)
_PHASE_AFTER = re.compile(r"after (?:the )?(?:promo|intro)\w*", re.I)
_GENERIC_LABEL = re.compile(
    r"^(rates?|select an option below:?|deposits?|loans?|table|total|yearly|submit|previous|next|"
    r"disclosure|contact us|home|menu|search|dupaco member)$",
    re.I,
)
# A first header that only names the kind of row ('Term', 'Account type') says nothing about the product.
_KEY_HEADER = re.compile(
    r"^(account type|certificate|ira type|product|program|loan program|loan type|card type|term|balance|"
    r"balance tier|loan amount|category|type|term length)$",
    re.I,
)
_FEE_URL = re.compile(r"fee", re.I)
_CELL_UNIT = re.compile(r"%\s*\*?\s*(APR|APY)\b", re.I)
_DITTO = ("″", '"', "do.", "ditto", "same")
MAX_PERCENT = 100  # a rate cell above this is a mis-read column
MIN_EARN_TIER_TOLERANCE = 0.011


def classify_header(h: str) -> tuple[str, str, dict] | None:
    """(attribute, numeric hint, column qualifiers) for a value column header, else None."""
    hl = h.lower().strip()
    if not hl:
        return None
    for rx, attr, hint in _COLS:
        if rx.search(hl):
            quals = {}
            m = _PHASE_INTRO.search(hl)
            if m:
                quals["phase"] = f"intro_{m.group(1)}mo" if m.group(1) else "intro"
            elif _PHASE_AFTER.search(hl):
                quals["phase"] = "after_intro"
            return attr, hint, quals
    return None


def informative_words(label: str) -> list[str]:
    """Words of a label that say something about the product (not 'rate', 'account', 'table', ...)."""
    return [w for w in canon_name(strip_segment(label)[0]).split() if w not in WEAK_TOKENS]


def _specificity(label: str) -> int:
    return sum(1 for w in informative_words(label) if w not in FAMILY_TOKENS)


def _pick_label(t: Table, footnote_label: str, header0: str) -> str:
    near_all = [l for l in reversed(short_label_lines(t.before)) if len(l) <= 70]
    ok = lambda l: informative_words(l) and not _GENERIC_LABEL.match(l.strip())
    near = [l for l in near_all if ok(l)]
    prev = [] if near_all else [l for l in reversed(short_label_lines(t.prev_lines)) if len(l) <= 70][:3]
    heads = [h for h in reversed(t.heading_path[1:]) if len(h) <= 70]
    seq = [l for l in near + ([header0] if header0 else []) + prev + heads if ok(l)]
    chosen = seq[0] if seq else (near_all[0] if near_all else (header0 or footnote_label or ""))
    if footnote_label and chosen != footnote_label and _specificity(chosen) == 0:
        a, b = set(informative_words(chosen)), set(informative_words(footnote_label))
        if {drop_final_s(w) for w in a} < {drop_final_s(w) for w in b}:
            chosen = footnote_label
    return chosen


def _first_effective(lines: list[str]) -> str | None:
    for l in lines:
        d = effective_dates(l)
        if d:
            return d[0]
    return None


def _business_page(url: str, header: str) -> bool:
    return is_business_url(url) or bool(header and strip_segment(header)[1])


def _is_name_or_tier_cell(c: str) -> bool:
    return bool(re.search(r"[A-Za-z]{3}|\$|\d", c)) and not re.fullmatch(r"\W*(n/?a|none|—|-|–)?\W*", c, re.I)


# ---------------------------------------------------------------------------------------------------- one table
@dataclass
class _Row:
    cells: list
    quals: dict
    residual: str
    group: str  # section label in force for the row
    product: str = ""
    family: str = ""
    conf: float = Conf.TABLE_ROW
    flags: list = field(default_factory=list)


@dataclass
class _Table:
    t: Table
    chunk: object
    index: int
    page: object
    vcols: dict
    dim_cols: list  # the other columns: they name the row
    fn_label: str
    header0: str
    label: str
    label_name: str
    label_biz: bool
    effective: str | None


def _value_columns(t: Table) -> dict:
    cols = {j: classify_header(h) for j, h in enumerate(t.headers)}
    return {j: c for j, c in cols.items() if c}


def _first_header(t: Table, vcols: dict) -> str:
    h = t.headers[0] if t.headers else ""
    named = (
        h
        and 0 not in vcols
        and not _GENERIC_LABEL.match(h)
        and _specificity(h) >= 1
        and not _KEY_HEADER.search(h.strip())
    )
    return h if named else ""


def _describe(t: Table, chunk, ti: int, page, vcols: dict) -> _Table:
    first_fn = next((t.header_fn[j] for j in vcols if t.header_fn[j]), "")
    fn_label = t.footnotes.get(first_fn, "") if first_fn else ""
    header0 = _first_header(t, vcols)
    label = _pick_label(t, fn_label, header0)
    label_name, label_biz = strip_segment(label)
    dim_cols = [j for j in range(len(t.headers)) if j not in vcols]
    effective = _first_effective(t.after) or _first_effective(t.before[-3:])
    return _Table(t, chunk, ti, page, vcols, dim_cols, fn_label, header0, label, label_name, label_biz, effective)


def _read_rows(tb: _Table) -> list[_Row]:
    rows: list[_Row] = []
    prev_label_cells: dict[int, str] = {}
    for ri, row in enumerate(tb.t.rows):
        cells = list(row)
        for j in tb.dim_cols[:1]:
            if (not cells[j] or cells[j] in _DITTO) and j in prev_label_cells:
                cells[j] = prev_label_cells[j]
            elif cells[j]:
                prev_label_cells[j] = cells[j]
        quals: dict = {}
        residuals: list[str] = []
        for d in (cells[j] for j in tb.dim_cols if cells[j] and _is_name_or_tier_cell(cells[j])):
            q, res = extract_qualifiers(d)
            for k, v in q.items():
                quals.setdefault(k, v)
            if res:
                residuals.append(res)
        rows.append(_Row(cells, quals, " ".join(residuals), tb.t.row_groups[ri]))
    return rows


def _name_row(row: _Row, tb: _Table) -> None:
    res_n, row_biz = strip_segment(row.residual)
    heading_path = tb.t.heading_path
    if row.quals.get("ira_type"):
        fam = "ira"
    else:
        fam = family_of(
            res_n, row.group, tb.label_name, tb.fn_label, tb.header0, *reversed(heading_path[1:]), tb.page.title
        )
    kind = "loan_amount" if fam in LOAN_FAMILIES else ("balance" if fam else "amount")
    q = {(f"{kind}_range" if k == "amount_range" else k): v for k, v in row.quals.items()}
    cname = canon_name(res_n)
    kname = canon_name(tb.label_name or row.group)
    selfdesc = sum(1 for k in ("purpose", "term_months", "loan_amount_range", "model_year", "arm") if k in q) >= 2
    if not cname:
        prod = kname or canon_name(row.group)
    elif selfdesc or (fam and has_family_keyword(cname, fam)) or not kname:
        prod = cname
    else:
        kt = kname.split()
        prod = " ".join(kt + [w for w in cname.split() if w not in kt])
    prod = finish_product(prod, fam)
    if row_biz or tb.label_biz or _business_page(tb.page.url, tb.header0):
        q["segment"] = "business"
    flags = []
    if not (cname and (selfdesc or (fam and has_family_keyword(cname, fam)))):
        flags.append("label_only")
    conf = Conf.TABLE_ROW
    if not prod:
        prod, conf = "unknown", Conf.TABLE_UNRESOLVED
        flags.append("product_unresolved")
    elif not cname and not informative_words(tb.label):
        conf = Conf.TABLE_GENERIC_LABEL
        flags.append("generic_label")
    if tb.page.parsed.junk_table:
        conf -= Conf.JUNK_TABLE_PENALTY
    row.quals, row.product, row.family, row.conf, row.flags = q, prod, fam, conf, flags


def _model_year_ranks(rows: list[_Row]) -> dict:
    bands = sorted(
        {r.quals["model_year"] for r in rows if "model_year" in r.quals},
        key=lambda b: (b[1] or 0, b[0] or 0),
        reverse=True,
    )
    return {b: i for i, b in enumerate(bands)}


def _constant_column(rows: list[_Row], vcols: dict, attr: str):
    for j, (a, hint, _) in vcols.items():
        if a == attr:
            vals = [r.cells[j] for r in rows if r.cells[j] and parse_value(r.cells[j], hint)]
            if len(vals) >= 2 and len(set(vals)) == 1:
                return j
    return None


class _Emitter:

    def __init__(self, tb: _Table, rows: list[_Row], stats):
        self.tb, self.rows, self.stats = tb, rows, stats
        self.rank = _model_year_ranks(rows)
        self.const_open = _constant_column(rows, tb.vcols, "min_to_open")
        self.emitted_const: set = set()

    def run(self) -> list[Fact]:
        out: list[Fact] = []
        for row in self.rows:
            for j, (attr, hint, col_quals) in self.tb.vcols.items():
                obs = self._cell(row, j, attr, hint, col_quals)
                if obs is not None:
                    out.append(obs)
        return out

    def _cell(self, row: _Row, j: int, attr: str, hint: str, col_quals: dict) -> Fact | None:
        stats, tb = self.stats, self.tb
        cell = row.cells[j]
        v = parse_value(cell, hint)
        if v is None:
            stats["cells_skipped_na"] += 1
            return None
        if (v.unit == "%" and (v.lo > MAX_PERCENT or (v.hi or 0) > MAX_PERCENT)) or (
                attr == "dividend_rate" and v.unit != "%"):
            stats["cells_skipped_implausible"] += 1
            return None
        fq = dict(row.quals)
        fq.update(col_quals)
        cu = _CELL_UNIT.search(cell)
        xflags: list[str] = []
        if cu and v.unit == "%" and attr in ("apy", "apr", "rate"):
            if cu.group(1).lower() != attr:
                xflags.append("unit_from_cell")
            attr = cu.group(1).lower()
        if attr == "min_to_earn":
            tier = fq.get("balance_range") or fq.get("loan_amount_range")
            if tier and tier[0] is not None and abs(tier[0] - v.lo) < MIN_EARN_TIER_TOLERANCE:
                stats["cells_skipped_redundant"] += 1
                return None
        if attr == "min_to_open" and self.const_open is not None and j == self.const_open:
            fq.pop("balance_range", None)
            sig = (row.product, tuple(sorted(fq.items())), v.lo)
            if sig in self.emitted_const:
                return None
            self.emitted_const.add(sig)
        loose = tuple(sorted(({**fq, "model_year": ("rank", self.rank[fq["model_year"]])} if "model_year" in fq
                              else fq).items()))
        flags = tuple(row.flags + xflags + [x for x in v.flags if not (x == "unit_from_header" and hint == "n")])
        conf = row.conf - (Conf.UNIT_FROM_HEADER_PENALTY if "unit_from_header" in flags else 0)
        return Fact.in_chunk(
            tb.page, tb.chunk, key=make_key(row.product, attr, fq),
            product_raw=(row.cells[tb.dim_cols[0]] if tb.dim_cols else "") or tb.label, family=row.family,
            value=v.lo, value_hi=v.hi, unit=v.unit, raw_value=cell, caption=tb.label, column=tb.t.headers[j],
            row_text=" | ".join(row.cells), effective_date=tb.effective,
            effective_scope="chunk" if tb.effective else None, source_kind="table",
            confidence=round(conf, 2), flags=flags, table_index=tb.index, loose_qualifiers=loose)


def _table_facts(t: Table, chunk, ti: int, page, stats) -> list[Fact]:
    vcols = _value_columns(t)
    if not vcols or not t.rows:
        stats["tables_skipped_no_value_column"] += 1
        stats.log.append(
            (
                "skipped_table",
                page.url,
                " | ".join(t.headers)[:100],
                any(re.search(r"%|\$\s?\d", c) for r in t.rows for c in r),
            )
        )
        return []
    tb = _describe(t, chunk, ti, page, vcols)
    rows = _read_rows(tb)
    stats["tables_used"] += 1
    for row in rows:
        _name_row(row, tb)
    facts = _Emitter(tb, rows, stats).run()
    if not facts:
        stats["tables_no_facts"] += 1
        stats.log.append(("no_facts_table", page.url, " | ".join(t.headers)[:100], True))
    return facts


def extract_tables(page, stats) -> list[Fact]:
    """Observations from every table of one page version."""
    out: list[Fact] = []
    chunks = page.parsed.chunks
    for i, ch in enumerate(chunks):
        if "\n|" not in "\n" + ch.text:
            continue
        prev = [c for c in chunks[max(0, i - 3) : i] if c.source_url == ch.source_url]
        for ti, t in enumerate(parse_tables(ch, prev)):
            stats["tables_seen"] += 1
            out.extend(_table_facts(t, ch, ti, page, stats))
    return out


def is_fee_schedule_pdf(page) -> bool:
    return page.parsed.fmt == "pdf" and bool(_FEE_URL.search(page.url))
