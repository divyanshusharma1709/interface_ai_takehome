"""The bi-temporal fact store: validity windows, histories, disagreements, product matching and fact queries."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..core.dates import parse_as_of
from .facts import (
    AUTHORITY,
    CONTACT_ATTRS,
    RATE_ATTRS,
    Conf,
    Disagreement,
    Fact,
    FactKey,
    Version,
    dims,
    make_key,
    qual_match,
    tier_dimension,
    tier_qual,
)
from .normalize import GENERIC_WORDS, QUERY_STOP, canon_name, family_noun, parse_query, range_width, tokens


# ---- timeline
@dataclass
class Timeline:
    active: dict[str, list[Fact]] = field(default_factory=dict)
    dups: dict[str, list[Fact]] = field(default_factory=dict)
    versions: dict[tuple, list[Version]] = field(default_factory=dict)  # (key, url) -> versions, oldest first
    by_key: dict[FactKey, list[Version]] = field(default_factory=dict)
    relabel: dict[FactKey, tuple[FactKey, str]] = field(default_factory=dict)


def build_timeline(raw: dict[str, list[Fact]], snap_dates: dict[str, str], pages_by_snap: dict[str, set]) -> Timeline:
    """`raw`: snapshot id -> every observation extracted from it."""
    sids = sorted(raw, key=lambda s: snap_dates[s])
    tl = Timeline()
    primary: dict[str, dict[tuple, Fact]] = {}
    for sid in sids:
        primary[sid], tl.dups[sid] = _choose_primaries(raw[sid])
        tl.active[sid] = list(primary[sid].values())
    keys = sorted({k for sid in sids for k in primary[sid]}, key=lambda k: (str(k[0]), k[1]))
    for kk in keys:
        vers = _key_versions(kk, sids, primary, snap_dates, pages_by_snap)
        tl.versions[kk] = vers
        tl.by_key.setdefault(kk[0], []).extend(vers)
    tl.relabel = _link_relabelled(tl.versions, sids)
    return tl


def _choose_primaries(facts: list[Fact]) -> tuple[dict, list]:
    groups: dict[tuple, list[Fact]] = defaultdict(list)
    for f in facts:
        groups[(f.key, f.url)].append(f)
    primary, dups = {}, []
    for k, fs in groups.items():
        if len(fs) == 1:
            primary[k] = fs[0]
            continue
        fs.sort(key=lambda f: (f.effective_date or "", f.confidence, -f.chunk_ordinal, -f.table_index), reverse=True)
        p = fs[0]
        for o in fs[1:]:
            if o.value_sig == p.value_sig:
                o.flags = tuple(sorted(set(o.flags) | {"duplicate"}))
            elif o.effective_date and p.effective_date and o.effective_date < p.effective_date:
                o.flags = tuple(sorted(set(o.flags) | {"superseded_in_page"}))
                p.alts.append(o)
            else:
                o.flags = tuple(sorted(set(o.flags) | {"ambiguous_in_page"}))
                p.alts.append(o)
            dups.append(o)
        primary[k] = p
    return primary, dups


def _runs(
    kk: tuple, sids: list, primary: dict, snap_dates: dict, pages_by_snap: dict
) -> list[tuple[list, str | None, str]]:
    url = kk[1]
    runs: list[list[Fact]] = []
    closed: dict[int, tuple[str | None, str]] = {}
    open_run: list[Fact] | None = None
    for sid in sids:
        f = primary[sid].get(kk)
        if f is None:
            if open_run is not None:
                gone = url not in pages_by_snap.get(sid, ())
                closed[len(runs) - 1] = (snap_dates[sid], "page_gone" if gone else "removed")
                open_run = None
        elif open_run is not None and open_run[-1].value_sig == f.value_sig:
            open_run.append(f)
        else:
            if open_run is not None:
                closed[len(runs) - 1] = ("", "changed")
            open_run = [f]
            runs.append(open_run)
    return [(run, *closed.get(i, (None, ""))) for i, run in enumerate(runs)]


def _key_versions(kk: tuple, sids: list, primary: dict, snap_dates: dict, pages_by_snap: dict) -> list[Version]:
    key, url = kk
    runs = _runs(kk, sids, primary, snap_dates, pages_by_snap)
    vers: list[Version] = []
    for i, (run, until, why) in enumerate(runs):
        first, last = run[0], run[-1]
        seen_on = min((first.fetched_at or first.taken_on)[:10], first.taken_on)
        eff = first.effective_date
        vf = eff if eff and eff <= max(seen_on, first.taken_on) else seen_on
        if why == "changed":
            nxt = runs[i + 1][0][0]
            nobs = min((nxt.fetched_at or nxt.taken_on)[:10], nxt.taken_on)
            until = (
                nxt.effective_date
                if nxt.effective_date and nxt.effective_date >= vf and nxt.effective_date <= nobs
                else nobs
            )
            until = max(until, vf)
        v = Version(
            key,
            url,
            first.value,
            first.value_hi,
            first.value_text,
            first.unit,
            first.snapshot_id,
            last.snapshot_id,
            vf,
            seen_on,
            until,
            why,
            last.page_type,
            last.effective_date,
            last.confidence,
            last,
            (url,),
            list(run),
        )
        vers.append(v)
        _set_window(v.facts, v.first_snapshot, v.last_snapshot, vf, seen_on, until, why)
    for a, b in zip(vers, vers[1:]):
        if a.valid_until is not None and b.valid_from < a.valid_until and b.valid_from >= a.valid_from:
            a.valid_until, a.ended_reason = b.valid_from, "changed"
            for f in a.facts:
                f.valid_until, f.ended_reason = b.valid_from, "changed"
    return vers


def _set_window(
    facts: list[Fact],
    first_snapshot: str,
    last_snapshot: str,
    valid_from: str,
    observed_from: str,
    valid_until,
    why: str,
) -> None:
    for f in facts:
        f.first_snapshot, f.last_snapshot, f.valid_from, f.observed_from = (
            first_snapshot,
            last_snapshot,
            valid_from,
            observed_from,
        )
        f.valid_until, f.ended_reason = valid_until, why


def _link_relabelled(versions: dict, sids: list[str]) -> dict:
    pos = {sid: i for i, sid in enumerate(sids)}
    by_slot: dict[tuple, list[Version]] = defaultdict(list)
    for (key, url), vs in versions.items():
        for v in vs:
            if v.facts[0].loose_qualifiers and any(n == "model_year" for n, _ in key.qualifiers):
                by_slot[(url, key.product, key.attribute)].append(v)
    relabel: dict[FactKey, tuple[FactKey, str]] = {}
    for vs in by_slot.values():
        for a in vs:
            if a.ended_reason != "removed" or a.valid_until is None:
                continue
            for b in vs:
                if (
                    b.key != a.key
                    and b.facts[0].loose_qualifiers == a.facts[-1].loose_qualifiers
                    and pos[b.first_snapshot] == pos[a.last_snapshot] + 1
                    and b.valid_from >= a.valid_from
                ):
                    until = min(a.valid_until, b.valid_from)
                    a.valid_until, a.ended_reason = until, "relabeled"
                    for f in a.facts:
                        f.valid_until, f.ended_reason = until, "relabeled"
                    relabel.setdefault(a.key, (b.key, until))
                    break
    return relabel


# ---- merged_history
def merged_history(
    key: FactKey,
    url: str | None,
    active: dict[str, list[Fact]],
    sids: list[str],
    snap_dates: dict[str, str],
    relabel: dict,
) -> list[Version]:
    """Runs of the same winning value become one `Version` listing all URLs that showed it."""
    segs: list[Version] = []
    cut = relabel[key][1] if key in relabel else None
    for sid in sids:
        if cut and snap_dates[sid] >= cut:
            break
        fs = [f for f in active[sid] if f.key == key and (url is None or f.url == url)]
        if fs:
            _add_snapshot(segs, key, sid, fs, snap_dates)
    if segs and segs[-1].last_snapshot != sids[-1]:
        segs[-1].valid_until, segs[-1].ended_reason = snap_dates[sids[-1]], "removed"
        if key in relabel:
            segs[-1].valid_until = relabel[key][1]
            segs[-1].ended_reason = "relabeled"
    return segs


def _add_snapshot(segs: list[Version], key: FactKey, sid: str, fs: list[Fact], snap_dates: dict[str, str]) -> None:
    w = max(fs, key=lambda f: (AUTHORITY.get(f.page_type, 1.0), f.effective_date or "", f.fetched_at, f.url))
    urls = tuple(sorted({f.url for f in fs if f.value_sig == w.value_sig}))
    if segs and segs[-1].fact.value_sig == w.value_sig:
        segs[-1].last_snapshot = sid
        segs[-1].urls = tuple(sorted(set(segs[-1].urls) | set(urls)))
        segs[-1].fact = w
        return
    if segs:
        # the change dates from the new value's effective date when that is later; a value that is merely older
        # (a stale page outliving the one that was current) takes over at the crawl that first lacks the winner
        segs[-1].valid_until = w.valid_from if w.valid_from > segs[-1].valid_from else snap_dates[sid]
        segs[-1].ended_reason = "changed"
    segs.append(
        Version(
            key,
            w.url,
            w.value,
            w.value_hi,
            w.value_text,
            w.unit,
            sid,
            sid,
            w.valid_from,
            w.observed_from,
            None,
            "",
            w.page_type,
            w.effective_date,
            w.confidence,
            w,
            urls,
        )
    )


# ---- disagreements
KINDS = ("cross_page", "within_page", "changed")


def find_disagreements(
    pool: list[Fact],
    snapshot_id: str,
    as_of: str | None,
    history: Callable[[FactKey], list[Version]],
    keys: Iterable[FactKey],
    *,
    include_prose: bool,
    min_confidence: float,
    kinds: Iterable[str],
    loose: bool,
) -> list[Disagreement]:
    """`pool`: the facts in effect on `as_of`; `history(key)`: the merged value history; `keys`."""
    if not pool:
        return []
    kinds = set(kinds)
    usable = lambda f: f.confidence >= min_confidence and (include_prose or f.source_kind != "prose")
    out: list[Disagreement] = []
    if "cross_page" in kinds:
        out += _cross_page(pool, snapshot_id, usable, loose)
    if "within_page" in kinds:
        out += _within_page(pool, snapshot_id, usable)
    if "changed" in kinds:
        out += _changed(keys, snapshot_id, as_of, history, min_confidence, include_prose)
    return out


def _cross_page(pool, sid, usable, loose) -> list[Disagreement]:
    group_key = (
        (lambda f: FactKey(f.product, f.attribute, f.loose_qualifiers or f.key.qualifiers))
        if loose
        else (lambda f: f.key)
    )
    groups: dict[FactKey, list[Fact]] = defaultdict(list)
    for f in pool:
        if usable(f):
            groups[group_key(f)].append(f)
    out = []
    for key, fs in sorted(groups.items(), key=lambda t: str(t[0])):
        vals = sorted({(f.value, f.value_hi) for f in fs if f.value is not None}, key=lambda v: (v[0], v[1] or 0))
        if len(vals) > 1:
            same_key = len({f.key for f in fs}) == 1
            out.append(Disagreement("cross_page" if same_key else "model_year_shift", key, sid, vals, fs))
    return out


def _within_page(pool, sid, usable) -> list[Disagreement]:
    out = []
    for f in pool:
        if usable(f) and any("ambiguous_in_page" in a.flags for a in f.alts):
            alts = [a for a in f.alts if "ambiguous_in_page" in a.flags and a.value_sig != f.value_sig and usable(a)]
            if alts:
                vals = sorted({(x.value, x.value_hi) for x in [f] + alts})
                out.append(Disagreement("within_page", f.key, sid, vals, [f] + alts))
    return out


def _changed(keys, sid, as_of, history, min_confidence, include_prose) -> list[Disagreement]:
    out = []
    for key in sorted(keys, key=str):
        vs = [
            v
            for v in history(key)
            if (as_of is None or v.valid_from <= as_of)
            and v.confidence >= min_confidence
            and (include_prose or v.fact.source_kind != "prose")
        ]
        if len(vs) > 1:
            changes = [
                (b.valid_from, (a.value, a.value_hi), (b.value, b.value_hi), b.urls)
                for a, b in zip(vs, vs[1:])
                if a.fact.value_sig != b.fact.value_sig
            ]
            if changes:
                out.append(Disagreement("changed", key, sid, [(v.value, v.value_hi) for v in vs], [], changes))
    return out


# ---- product_index
CREDIT_CARD_WORDS = {"credit", "card"}


class ProductIndex:
    """Built from the primary facts of every snapshot, oldest first."""

    def __init__(self, facts_by_snapshot: Iterable[list[Fact]]):
        fam: dict[str, Counter] = defaultdict(Counter)
        self.dims: dict[str, set] = defaultdict(set)
        for facts in facts_by_snapshot:
            for f in facts:
                fam[f.product][f.family] += 1
                if f.attribute in RATE_ATTRS:
                    self.dims[f.product] |= dims(f.qualifiers)
        self.family = {p: c.most_common(1)[0][0] for p, c in fam.items()}
        self.name: dict[str, frozenset] = {}  # tokens of the product name
        self.full: dict[str, frozenset] = {}  # ... plus its family noun's tokens
        for p, fm in self.family.items():
            name = frozenset(t for t in tokens(p) if t not in QUERY_STOP)
            extra = set(tokens(family_noun(fm))) | (CREDIT_CARD_WORDS if fm == "credit_card" else set())
            self.name[p], self.full[p] = name, name | extra

    def match(
        self, text: str, family: str | None = None, min_score: float = 0.5, parsed=None
    ) -> list[tuple[float, str]]:
        """Products matching free text, best first, as (score, product)."""
        pq = parsed or parse_query(text)
        fam = family or pq.family
        if pq.fee and not family:
            fam = "fee"
        words = list(dict.fromkeys(pq.words))
        pool = [p for p in self.family if not fam or self.family[p] == fam]
        generic = (
            set(GENERIC_WORDS) | set(tokens(family_noun(fam))) | (CREDIT_CARD_WORDS if fam == "credit_card" else set())
        )
        if pq.fee:
            words = [w for w in words if w not in generic]
        if not words:
            return [(1.0, p) for p in sorted(pool)]
        vocab = set().union(*(self.full[p] for p in pool)) if pool else set()
        known = [w for w in words if w in vocab]
        distinct = [w for w in known if w not in generic]
        rows = []
        for p in pool:
            full = self.full[p]
            hit = [w for w in known if w in full]
            if hit:
                rows.append(
                    (
                        sum(1 for w in distinct if w in full),
                        len(hit) / len(words),
                        len(hit) / max(1, len(self.name[p])),
                        p,
                    )
                )
        need = len(distinct) if any(r[0] == len(distinct) for r in rows) else max((r[0] for r in rows), default=0)
        out = [
            r for r in rows if r[0] >= need and r[1] + 1e-9 >= min_score
        ]  # all distinct words, else the best-covered
        out.sort(key=lambda t: (-t[1], -t[2], t[3]))
        return [(sc, p) for _, sc, _, p in out]


# ---- fact_query
# Words in the question that name the attribute asked for when the caller did not.
_ATTR_WORDS = [
    (re.compile(r"\brouting\b"), "routing_number"),
    (re.compile(r"\bapy\b|annual percentage yield"), "apy"),
    (re.compile(r"\bapr\b|annual percentage rate"), "apr"),
    (re.compile(r"\bdividend\b"), "dividend_rate"),
    (re.compile(r"\bpoints?\b"), "points"),
    (re.compile(r"\bmin\w*\.?\s+(?:balance\s+|deposit\s+)?to\s+open\b|\bopening deposit\b"), "min_to_open"),
    (re.compile(r"\bmin\w*\.?\s+(?:balance\s+)?to\s+earn\b"), "min_to_earn"),
    (re.compile(r"\bpayment example\b|\bmonthly payment\b"), "payment_example"),
]


@dataclass(frozen=True)
class FactQuery:
    product: str | None = None
    attribute: str | None = None
    qualifiers: dict | None = None
    as_of: object = None
    url: str | None = None
    family: str | None = None
    source_kind: str | None = None
    min_confidence: float | None = None
    fuzzy: bool = True
    loose: bool = False
    include_duplicates: bool = False
    segment: str | None = "auto"
    include_prose: bool = False
    min_score: float = 0.5
    one_tier: bool = True
    collapse: bool = False


class _Plan:

    def __init__(self, q: FactQuery, index: ProductIndex):
        self.q = q
        self.parsed = parse_query(q.product) if q.product else None
        self.explicit = dict(q.qualifiers or {})
        pq = self.parsed
        self.derived = {k: v for k, v in (pq.qualifiers.items() if pq else ()) if k not in self.explicit}
        self.attr_hint = None
        if pq and not q.attribute:
            self.attr_hint = "fee" if pq.fee else next((a for rx, a in _ATTR_WORDS if rx.search(pq.text)), None)
        self.want = self._wanted_products(index)
        self.nothing = self.want == {} and self.attr_hint not in CONTACT_ATTRS
        self.want_dims = set().union(*(index.dims.get(p, set()) for p in self.want)) if self.want else set()
        self.auto_segment = q.segment == "auto"
        self.segment = (pq.segment if pq and pq.segment else "personal") if self.auto_segment else q.segment
        self.min_confidence = q.min_confidence
        if self.min_confidence is None:
            self.min_confidence = 0.0 if (q.include_prose or q.source_kind == "prose") else Conf.ANSWER_MIN
        self.attribute = q.attribute or self.attr_hint

    def _wanted_products(self, index: ProductIndex) -> dict | None:
        q, pq = self.q, self.parsed
        if not q.product:
            return None
        if not q.fuzzy:
            return {p: 1.0 for p in {q.product, canon_name(q.product)}}
        if pq.words or pq.fee or q.family:
            want = {p: sc for sc, p in index.match(q.product, q.family, q.min_score, pq)}
            return want if want or self.attr_hint not in CONTACT_ATTRS else None
        return None

    def accepts(self, f: Fact, segment: str | None, attribute: str | None, index: ProductIndex) -> bool:
        q = self.q
        if self.want is not None and f.product not in self.want:
            return False
        if segment and f.qualifiers.get("segment", "personal") != segment:
            return False
        if attribute and f.attribute != attribute:
            return False
        if q.family and f.family != q.family:
            return False
        if q.url and q.url not in (f.url, f.raw_url):
            return False
        if q.source_kind and f.source_kind != q.source_kind:
            return False
        if f.confidence < self.min_confidence and f.attribute not in CONTACT_ATTRS:
            return False
        fq = f.qualifiers
        if self.explicit and not qual_match(fq, self.explicit, q.loose):
            return False
        if self.derived and not qual_match(fq, self.derived, True):
            return False
        return not (self.derived and f.attribute in RATE_ATTRS and self._lacks_dimension(f, fq, index))

    def _lacks_dimension(self, f: Fact, fq: dict, index: ProductIndex) -> bool:
        need = self.want_dims if self.want else index.dims.get(f.product, set())
        have = dims(fq)
        return any(tier_dimension(n) in need and tier_dimension(n) not in have for n in self.derived)


def _tightest_tiers(facts: list[Fact]) -> list[Fact]:
    best: dict[tuple, Fact] = {}
    keep = []
    for f in facts:
        tq = tier_qual(f.qualifiers)
        if tq is None:
            keep.append(f)
            continue
        rest = tuple((k, v) for k, v in f.key.qualifiers if k != tq[0])
        gk = (f.product, f.attribute, f.url, rest)
        cur = best.get(gk)
        if cur is None or range_width(tq[1]) < range_width(tier_qual(cur.qualifiers)[1]):
            best[gk] = f
    return keep + list(best.values())


def _newest_first(f: Fact) -> str:
    return "".join(chr(255 - ord(c)) for c in (f.effective_date or ""))  # descending date as a sort key


def select_facts(q: FactQuery, pool: list[Fact], index: ProductIndex) -> list[Fact]:
    """The facts of `pool` (the primary facts in effect, plus duplicates when asked) that answer `q`."""
    plan = _Plan(q, index)
    if plan.nothing:
        return []
    pick = lambda segment, attribute: [f for f in pool if plan.accepts(f, segment, attribute, index)]
    out = pick(plan.segment, plan.attribute)
    if not out and plan.attr_hint and not q.attribute:
        out = pick(plan.segment, None)
    if not out and plan.auto_segment:
        out = pick(None, plan.attribute) or (pick(None, None) if plan.attr_hint and not q.attribute else [])
    ex = plan.explicit
    amount = ex.get("amount", ex.get("balance", ex.get("loan_amount", plan.derived.get("amount"))))
    if q.one_tier and isinstance(amount, (int, float)):
        out = _tightest_tiers(out)
    order = {p: i for i, p in enumerate(plan.want or {})}
    out.sort(
        key=lambda f: (
            order.get(f.product, 0),
            f.product,
            f.attribute,
            -AUTHORITY.get(f.page_type, 1.0),
            _newest_first(f),
            str(f.key.qualifiers),
            f.url,
        )
    )
    if q.collapse:
        seen, uniq = set(), []
        for f in out:
            if f.key not in seen:
                seen.add(f.key)
                uniq.append(f)
        out = uniq
    return [copy.copy(f) for f in out]


# ---- store
class FactStore:
    """Bi-temporal store of numeric facts extracted from the snapshots."""

    def __init__(self) -> None:
        self.raw: dict[str, list[Fact]] = {}  # snapshot id -> every extracted observation
        self.snap_dates: dict[str, str] = {}  # snapshot id -> taken_on
        self._pages_by_snap: dict[str, set[str]] = {}
        self._timeline: Timeline | None = None
        self._products: ProductIndex | None = None

    def add_snapshot(self, view, observations: list[Fact]) -> None:
        """Register one snapshot: its date, its page set and the observations extracted from it."""
        sid = view.snapshot_id
        self.snap_dates[sid] = view.taken_on
        self._pages_by_snap[sid] = set(view.pages)
        self.raw[sid] = observations
        self._timeline = self._products = None

    def snapshot_ids(self) -> list[str]:
        return sorted(self.raw, key=lambda s: self.snap_dates[s])

    def _build(self) -> Timeline:
        if self._timeline is None:
            self._timeline = build_timeline(self.raw, self.snap_dates, self._pages_by_snap)
            sids = self.snapshot_ids()
            self._products = ProductIndex(self._timeline.active[s] for s in sids)
        return self._timeline

    # The built state as the answer and conflict layers read it.
    @property
    def _active(self) -> dict[str, list[Fact]]:
        return self._build().active

    @property
    def products(self) -> ProductIndex:
        self._build()
        return self._products

    @property
    def _pfull(self) -> dict[str, frozenset]:
        return self.products.full

    @property
    def _pfamily(self) -> dict[str, str]:
        return self.products.family

    def _snapshot_for(self, as_of) -> str | None:
        sids = self.snapshot_ids()
        if not sids:
            return None
        d = parse_as_of(as_of)
        if d is None:
            return sids[-1]
        ok = [s for s in sids if self.snap_dates[s] <= d]
        return ok[-1] if ok else None

    def _at(self, as_of=None) -> list[Fact]:
        tl = self._build()
        sids = self.snapshot_ids()
        if not sids:
            return []
        d = parse_as_of(as_of)
        if d is None:
            return list(tl.active[sids[-1]])
        out: list[Fact] = []
        for vs in tl.versions.values():
            for v in vs:
                if v.valid_from <= d and (v.valid_until is None or d < v.valid_until):
                    seen = [f for f in v.facts if f.taken_on <= d]
                    out.append(seen[-1] if seen else v.facts[0])
        return out

    def view(self, as_of=None) -> list[Fact]:
        """The facts the website showed on `as_of`: the primary facts of the latest snapshot taken on or before."""
        active = self._build().active
        sids = self.snapshot_ids()
        if not sids:
            return []
        d = parse_as_of(as_of)
        if d is None:
            return list(active[sids[-1]])
        sid = self._snapshot_for(as_of)
        if sid is not None:
            return list(active[sid])
        return [f for f in active[sids[0]] if f.valid_from and f.valid_from <= d]

    def history_at(self, key, as_of=None) -> list[Version]:
        """Merged value history of a key as visible on `as_of`."""
        hist = self.history(key, merged=True)
        d = parse_as_of(as_of)
        if d is None:
            return hist
        sid = self._snapshot_for(as_of)
        if sid is None:
            first_day = self.snap_dates[self.snapshot_ids()[0]]
            return [
                h for h in hist if h.valid_from and h.valid_from <= d and self.snap_dates[h.first_snapshot] == first_day
            ]
        cut = self.snap_dates[sid]
        return [h for h in hist if self.snap_dates[h.first_snapshot] <= cut]

    def facts(
        self,
        product: str | None = None,
        attribute: str | None = None,
        qualifiers: dict | None = None,
        as_of=None,
        **options,
    ) -> list[Fact]:
        """Facts in effect on `as_of` (None = latest snapshot), best match first."""
        return self.query(FactQuery(product, attribute, qualifiers, as_of, **options))

    def query(self, q: FactQuery) -> list[Fact]:
        tl = self._build()
        sid = self._snapshot_for(q.as_of)
        pool = self.view(q.as_of) + (tl.dups[sid] if q.include_duplicates and sid else [])
        return select_facts(q, pool, self.products)

    @staticmethod
    def _as_key(key) -> FactKey:
        if isinstance(key, FactKey):
            return key
        product, attribute, quals = key
        return make_key(product, attribute, dict(quals))

    def versions(self, key, url: str | None = None) -> list[Version]:
        """Per-URL versions of a key (value runs with validity windows), oldest first."""
        tl = self._build()
        vs = [v for v in tl.by_key.get(self._as_key(key), []) if url in (None, v.url)]
        return sorted(vs, key=lambda v: (v.valid_from, v.url))

    def history(self, key, url: str | None = None, merged: bool = False) -> list[Version]:
        """Ordered value changes of a key."""
        if not merged:
            return self.versions(key, url)
        tl = self._build()
        return merged_history(self._as_key(key), url, tl.active, self.snapshot_ids(), self.snap_dates, tl.relabel)

    def disagreements(
        self,
        as_of=None,
        include_prose: bool = False,
        min_confidence: float = Conf.ANSWER_MIN,
        kinds: Iterable[str] = KINDS,
        loose: bool = False,
    ) -> list[Disagreement]:
        """Keys that carry more than one distinct value (see `find_disagreements`)."""
        tl = self._build()
        sid = self._snapshot_for(as_of) or (self.snapshot_ids() or [""])[0]
        return find_disagreements(
            self._at(as_of),
            sid,
            parse_as_of(as_of),
            lambda k: self.history(k, merged=True),
            tl.by_key,
            include_prose=include_prose,
            min_confidence=min_confidence,
            kinds=kinds,
            loose=loose,
        )
