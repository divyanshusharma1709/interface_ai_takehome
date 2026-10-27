"""Rate conflicts from the fact store: different rates for one product, and IRA rows that should move together."""
from __future__ import annotations

from collections import defaultdict

from ..ingest.fact_format import attr_name, display_product, display_value, qual_phrases
from ..ingest.fact_format import authority, rank_key
from ..ingest.facts import AUTHORITY
from . import vocab as V
from .base import MAX_SOURCES, Cand, Miner, RateInfo, fact_source, path_of

ATTR_ORDER = {a: i for i, a in enumerate(V.RATE_ATTRS)}
NO_TIER = (False, 0, 1e18)         # ordering key of a row without a tier; 1e18 sorts an open-ended tier last


class RateFacts:
    """The fact store's per-snapshot views; the only code that reaches into its snapshot-building internals."""

    def __init__(self, store):
        store._build()
        self.store = store
        self.sids = store.snapshot_ids()

    def cross_page(self, sid):
        return self.store.disagreements(as_of=self.store.snap_dates[sid], kinds=("cross_page",), loose=True,
            min_confidence=0.4)

    def primary(self, sid):
        return self.store._active[sid]

    def in_effect(self, sid):
        return self.store._at(self.store.snap_dates[sid])


def term_label(months):
    return f"{months // 12}-year" if months >= 120 and months % 12 == 0 else f"{months}-month"


def tier_sort(f):
    # Headline row first: purchase before refinance, newest model-year band, lowest tier.
    q = f.key.qdict
    tier = next((q[n] for n in ("loan_amount_range", "balance_range", "amount_range") if isinstance(q.get(n), tuple)),
        None)
    my = q.get("model_year")
    return (str(q.get("purpose", "")) != "purchase", -((my[1] or 0) if isinstance(my, tuple) else (my or 0)),
            tier[1] if tier and tier[1] is not None else NO_TIER[2])


def differs(a, b):
    return (a.value, a.value_hi) != (b.value, b.value_hi)


class _Row:
    """One disagreement of one snapshot: the winning reading of a fact key and the readings it beat."""

    def __init__(self, key, winner, losers):
        self.key, self.winner, self.losers = key, winner, losers


class NumericMiner(Miner):
    def run(self):
        facts = RateFacts(self.kb.facts)
        # fact key -> snapshot -> rows: one conflict per specific fact
        clusters = defaultdict(lambda: defaultdict(list))
        for sid in facts.sids:
            for x in facts.cross_page(sid):
                valued = [f for f in x.facts if f.value is not None]
                if x.key.attribute in V.RATE_ATTRS and valued:
                    winner = max(valued, key=rank_key)
                    losers = [f for f in valued if differs(f, winner)]
                    if losers:
                        clusters[str(x.key)][sid].append(_Row(x.key, winner, losers))
        self._add_same_page(clusters, facts)
        last = facts.sids[-1] if facts.sids else None
        return [self._candidate(ck, by_sid, last) for ck, by_sid in sorted(clusters.items(), key=lambda t: str(t[0]))]

    def _add_same_page(self, clusters, facts):
        # A key stated twice on one page (a stale table under a fresh one) joins the cross-page rows of that
        # fact.
        for sid in facts.sids:
            extra = {}
            for p in facts.primary(sid):
                if p.key.attribute not in V.RATE_ATTRS or p.value is None:
                    continue
                for a in p.alts:
                    ambiguous = "ambiguous_in_page" in a.flags and "prose" not in (p.source_kind, a.source_kind)
                    if a.value is None or not differs(a, p) or not ("superseded_in_page" in a.flags or ambiguous):
                        continue
                    extra.setdefault(str(p.key), _Row(p.key, p, [])).losers.append(a)
            for ck, row in extra.items():
                clusters[ck + "#page"][sid].append(row)

    def _candidate(self, ck, by_sid, last):
        sid = max(by_sid)
        persistent = sid == last
        rows = sorted(by_sid[sid], key=lambda r: (ATTR_ORDER[r.key.attribute], tier_sort(r.winner), str(r.key)))
        head = rows[0]
        w = head.winner
        first_key = by_sid[sid][0].key          # the unsorted first row names the product and qualifiers
        q0 = first_key.qdict
        business = q0.get("segment", "personal") == "business"
        core, att, topic = self._topic(ck, rows, w, first_key)
        lo0 = sorted(head.losers, key=rank_key, reverse=True)[0]
        stale_urls = {lo.url for r in rows for lo in r.losers}
        where = ("an older table on the same page" if lo0.url == w.url else
                 f"{len(stale_urls)} other page(s) {'still show' if persistent else 'showed'} an older value, e.g. "
                 f"{path_of(lo0.url)}")
        reason = (f"Winner {display_value(w)} from {w.page_type.replace('_', ' ')} {path_of(w.url)} (effective "
                  f"{w.effective_date or 'n/a'}, crawled {w.fetched_at[:10]}); {where} {display_value(lo0)} "
                  f"({lo0.page_type.replace('_', ' ')}, effective {lo0.effective_date or 'n/a'}). "
                  f"Rule: {' + '.join(self._why(w, lo0))} (same ranking "
                  f"as answer()).") + self._competing_notes(head.losers)
        if not persistent:
            reason += f" The pages agreed again by the latest snapshot (last conflict in snapshot {sid[:2]})."
        gap = authority(w) - authority(lo0)
        tie = (w.effective_date or "") == (lo0.effective_date or "") and gap <= 0
        # the pages still disagree in the newest snapshot / agreed again by then
        conf = 0.92 if persistent else 0.8
        conf -= 0.2 if tie else 0                  # an equal-authority tie
        conf -= 0.2 if {w.page_type, lo0.page_type} & {"blog", "newsroom"} else 0        # one side is a story
        sal = V.FAMILY_SALIENCE.get(w.family, 0.6) * (0.85 if business else 1.0) * (1 + 0.04 * min(len(stale_urls), 6))
        fold = (sid[:2], (first_key.product, tuple((k, v) for k, v in w.key.qualifiers if k != "purpose"),
                          ck.endswith("#page")))
        order = (0 if persistent else 1, int(lo0.url == w.url), int(business), int(V.JUMBO_WORD in core.lower()),
                 tier_sort(w))
        info = RateInfo(fold, head.key.attribute, q0.get("purpose"), sorted({lo.url for lo in head.losers}), core,
                        topic[len(core) + 1 + len(att):], order)
        return Cand("numeric", topic, self._values(rows), display_value(w), reason, self._sources(w, head),
                    round(conf, 3), round(sal, 3), f"num:{sid[:2]}:{ck}", rate=info)

    @staticmethod
    def _values(rows):
        # The first three distinct values: each row's winner, then its losers.
        values = []
        for r in rows:
            for f in [r.winner] + sorted(r.losers, key=lambda f: (f.value, f.url)):
                v = display_value(f)
                if v not in values and len(values) < 3:
                    values.append(v)
        return values

    @staticmethod
    def _topic(ck, rows, w, first_key):
        q0 = first_key.qdict
        term, ira = q0.get("term_months", 0), str(q0.get("ira_type", ""))
        pname = display_product(first_key.product, w.family)
        fam_word = V.FAMILY_WORD.get(w.family, "")
        core = pname if (not fam_word or fam_word.split()[0] in pname.lower()) else f"{pname} {fam_word}"
        att = " / ".join(attr_name(a) for a in sorted({r.key.attribute for r in rows}, key=ATTR_ORDER.get))
        bits = ([f"{term_label(int(term))} term"] if term else []) + ([f"{ira} IRA"] if ira else []) \
            + (["business"] if q0.get("segment", "personal") == "business" else []) \
            + qual_phrases({k: v for k, v in w.key.qdict.items() if k not in ("segment", "ira_type", "term_months")},
                w.family)
        topic = f"{core} {att}" + (f" ({', '.join(bits)})" if bits else "")
        return core, att, topic + (" - older table on the same page" if ck.endswith("#page") else "")

    @staticmethod
    def _why(w, lo0):
        why = []
        if (w.effective_date or "") > (lo0.effective_date or ""):
            why.append("newer effective date")
        if authority(w) - authority(lo0) > 0:
            why.append("more authoritative page (rate sheet / product page over a landing, promo or calculator page)")
        if "inverted_yield" in lo0.flags:
            why.insert(0, "the competing row states a dividend rate above its own APY, which is impossible")
        return why or ["best topical fit and most recent crawl"]

    @staticmethod
    def _competing_notes(losers):
        # One sentence per page (at most 3) whose free-text mention disagrees, saying why it does not count.
        notes, seen = "", set()
        for lo in sorted(losers, key=lambda f: (f.url, f.value)):
            if lo.source_kind not in ("line", "sentence") or lo.url in seen or len(seen) >= 3:
                continue
            seen.add(lo.url)
            kind, where, v = lo.page_type.replace("_", " "), path_of(lo.url), display_value(lo)
            if "expired_offer" in lo.flags:
                notes += (f" The competing {v} on {where} is a promotion that ended {lo.offer_end}, "
                          "so it no longer applies.")
            elif lo.page_type in ("blog", "newsroom"):
                notes += (f" The competing {v} is from a {kind} ({where}) published {lo.effective_date or 'earlier'}; "
                          f"the rate sheet is the current source.")
            elif "promo" in lo.flags or "from_rate" in lo.flags:
                notes += f" The competing {v} is promotional copy on {where}, not the product's rate table."
            else:
                notes += f" The competing {v} is free text on a {kind} ({where}), not the product's rate table."
        return notes

    @staticmethod
    def _sources(w, head):
        srcs, seen = [fact_source(w)], {w.url}
        for lo in sorted(head.losers, key=lambda f: (-AUTHORITY.get(f.page_type, 1.0), f.url)):
            if lo.url not in seen and len(srcs) < MAX_SOURCES:
                seen.add(lo.url)
                srcs.append(fact_source(lo))
        return srcs


class ParallelMiner(Miner):
    """Rows of one table that should move together but do not: the same special / term / tier for two IRA types."""

    def run(self):
        facts = RateFacts(self.kb.facts)
        found = {}
        for sid in facts.sids:
            for k, row in self._disagreeing_rows(facts, sid).items():
                found.setdefault(k[:3], {})[sid] = row
        return [self._candidate(gk, by_sid, facts.sids) for gk, by_sid in sorted(found.items())]

    @staticmethod
    def _disagreeing_rows(facts, sid):
        # Row key -> ({ira type: fact}, the same row of a plain certificate) where the IRA types state
        # different values.
        grouped, plain = defaultdict(lambda: defaultdict(list)), {}
        for f in facts.in_effect(sid):
            if f.value is None or f.key.attribute not in V.RATE_ATTRS:
                continue
            q = dict(f.key.qdict)
            ira = q.pop("ira_type", None)
            k = (" ".join(sorted(set(f.key.product.split()) - V.IRA_ROW_FILLER)), f.key.attribute,
                 str(sorted((a, str(b)) for a, b in q.items())), f.url)
            if ira:
                grouped[k][str(ira)].append(f)
            elif f.family == "certificate":
                plain[k] = f
        rows = {k: ({t: fl[0] for t, fl in by_type.items()}, plain.get(k)) for k,
            by_type in grouped.items() if len(by_type) >= 2}
        return {k: row for k, row in rows.items() if len({v.value for v in row[0].values()}) >= 2}

    @staticmethod
    def _candidate(gk, by_sid, sids):
        sid = max(by_sid)
        vals, sibling = by_sid[sid]
        types = sorted(vals)
        fa, fb = vals[types[0]], vals[types[1]]
        resolution = display_value(sibling) if sibling is not None and sibling.value in (fa.value, fb.value) else None
        pname = display_product(fa.key.product.replace(" ira", "").replace("special ", "") + " special", fa.family)
        q = {k: v for k, v in fa.key.qdict.items() if k != "ira_type"}
        term = q.pop("term_months", 0)
        bits = ([f"{term_label(int(term))} term"] if term else []) + qual_phrases(q,
            fa.family) + [V.IRA_TYPE_PAIR_LABEL]
        verdict = (f"The matching non-IRA certificate row shows {display_value(sibling)}, so that value is taken "
                   f"as correct."
                   if resolution else "No sibling row confirms either figure, so no winner is chosen.")
        reason = (f"The {types[0]} and {types[1]} IRA rows of the same special and term differ ({display_value(fa)} vs "
                  f"{display_value(fb)}) in snapshot {sid[:2]} although they are one product. " + verdict)
        return Cand("numeric", f"{pname} IRA {attr_name(fa.key.attribute)} ({', '.join(bits)})", [display_value(fa),
            display_value(fb)],
                    resolution, reason, [fact_source(fa), fact_source(fb)], 0.5 if sid != sids[-1] else 0.75,
                    V.FAMILY_SALIENCE.get(fa.family, 0.6), f"par:{gk}")
