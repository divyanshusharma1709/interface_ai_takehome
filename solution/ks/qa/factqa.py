"""Answering numeric / product questions from the fact store, and the words for the chosen rows."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from ..core.dates import month_day_year
from ..core.types import Result, SourceRef
from ..core.urls import normalize_url
from ..ingest.fact_format import (
    attr_name,
    authority,
    display_product,
    display_value,
    money,
    qual_phrases,
    rank_key,
    url_fit,
)
from ..ingest.normalize import parse_query
from ..site_profile import PRODUCT_ALIASES
from .fact_pick import PAIRED, FactMatcher, product_text
from .fact_rows import TIER_NAMES, FactLists, HubFinder, Row, build_rows, prune_rows

# ---- Words for the chosen fact rows: values, 'as of' dates, yes/no, 'highest' and 'changed' forms.
# the figure a row is compared by, in order
_PRIMARY_ATTRS = ("apy", "apr", "rate", "dividend_rate", "fee", "min_to_open", "min_to_earn")
_SOURCES_MAX = 3
_TIE = 1e-9
_VALUE_MATCH = 1e-6
_STILL = re.compile(r"\b(still|anymore|any more|no longer|now|currently)\b", re.I)


class FactRenderer:
    def __init__(self, kb):
        self.kb = kb
        self.fs = kb.facts

    # ------------------------------------------------------------------ one row
    def label(self, r: Row) -> str:
        name = display_product(r.product, r.family)
        if r.segment == "business" and not r.merged_segments and "business" not in name.lower():
            name = "Business " + name
        phrases = qual_phrases({k: v for k, v in r.quals.items() if v is not None}, r.family)
        if r.larger_tiers_to is not None and r.larger_tiers_to[1] is not None:
            phrases = [
                (
                    p + f" and larger tiers up to {money(r.larger_tiers_to[1])}"
                    if p.startswith(("loan amount", "balance"))
                    else p
                )
                for p in phrases
            ]
        if r.both_ira_types:
            phrases.insert(0, "Roth and Traditional IRA")
        return name + (f" ({', '.join(phrases)})" if phrases else "")

    @staticmethod
    def value_phrase(r: Row, q) -> str:
        # an explicit exclusion ('not the APY') also drops that figure from a whole-row rendering
        v = {a: f for a, f in r.vals.items() if a not in q.exclude_attrs} or r.vals
        if q.kind == "fee":
            f = next(iter(v.values()))
            unit = str(f.qualifiers.get("unit", "")).replace("_", " ")
            return display_value(f) + (f" {unit}" if unit else "")
        if q.kind in ("min_open", "min_earn"):
            return display_value(next(iter(v.values())))
        parts = []
        if "rate" in v:
            parts.append(f"rate {display_value(v['rate'])}")
        if "dividend_rate" in v:
            parts.append(f"dividend rate {display_value(v['dividend_rate'])}")
        if "apy" in v:
            parts.append(f"APY {display_value(v['apy'])}")
        if "apr" in v:
            parts.append(f"APR {display_value(v['apr'])}")
        if "rate" in v and "apr" in v:
            return f"{parts[0]} (APR {display_value(v['apr'])})"
        if "dividend_rate" in v and "apy" in v:
            return f"{parts[0]} (APY {display_value(v['apy'])})"
        return ", ".join(parts)

    @staticmethod
    def primary_value(r: Row) -> float | None:
        for a in _PRIMARY_ATTRS:
            if a in r.vals:
                return r.vals[a].value
        return None

    def line(self, r: Row, q) -> str:
        return f"{self.label(r)}: {self.value_phrase(r, q)}"

    # ------------------------------------------------------------------ the answer
    def render(self, q, rows: list[Row], pool) -> Result | None:
        rows = [r for r in rows if r.vals]
        if not rows:
            return None
        if q.intent == "change":
            res = self._change(q, rows)
            if res:
                return res
        if q.superlative and len(rows) > 1:
            rows = self._superlative(q, rows)
        lead = f"As of {month_day_year(q.as_of)}, " if q.as_of else ""
        if q.intent == "check":
            res = self._check(q, rows, lead)
            if res:
                return res
        lines = [self.line(r, q) for r in rows]
        if len(lines) == 1:
            body = (lead + lines[0][0].lower() + lines[0][1:] if lead else lines[0]) + "."
        else:
            body = (
                (
                    lead + f"here is what {self.kb.voice.label()} lists:"
                    if lead
                    else f"Here is what {self.kb.voice.label()} lists:"
                )
                + "\n"
                + "\n".join(f"- {l}" for l in lines)
            )
            if (
                q.kind == "rate"
                and any(isinstance(r.quals.get(n), tuple) for r in rows for n in TIER_NAMES)
                and not q.qualifiers.get("amount")
            ):
                body += "\nThe rate depends on the tier shown on each line."
        return Result(body, self.sources(rows, q), True, self.conflict_note(q, rows))

    def _superlative(self, q, rows: list[Row]) -> list[Row]:
        vals = [(self.primary_value(r), r) for r in rows if self.primary_value(r) is not None]
        if not vals:
            return rows
        pick = max(vals, key=lambda t: t[0]) if q.superlative == "high" else min(vals, key=lambda t: t[0])
        # on a tie prefer the row with the tighter (higher) tier, which is where a top rate starts
        ties = [r for v, r in vals if abs(v - pick[0]) < _TIE]
        return [ties[-1]] if len(ties) > 1 and q.superlative == "high" else [pick[1]]

    # ------------------------------------------------------------------ yes / no
    def _matching(self, q, rows: list[Row], vals: list, want) -> list:
        """(value, row) pairs that satisfy the quoted figure and comparison."""
        if q.compare == "gt" and want is not None:
            return [(v, r) for v, r in vals if v > want + _TIE]
        if q.compare == "lt" and want is not None:
            return [(v, r) for v, r in vals if v < want - _TIE]
        return [
            (f.value, r)
            for r in rows
            for f in r.vals.values()
            if want is not None and f.value is not None and abs(f.value - want) < _VALUE_MATCH
        ]

    def _was_true_once(self, q, rows: list[Row], want) -> Row | None:
        """The row that carried the quoted figure in an earlier snapshot (the question says 'still' / 'now')."""
        past = [
            r
            for r in rows
            if any(
                abs(h.fact.value - want) < _VALUE_MATCH
                for f in r.vals.values()
                for h in self.fs.history_at(f.key, q.as_of)
                if h.fact.value is not None
            )
        ]
        return past[0] if past else None

    def _check(self, q, rows: list[Row], lead: str) -> Result | None:
        """Yes/no questions about a quoted value ('does any X pay 3%?', 'is it still 4.10%?')."""
        want = q.numbers[0] if q.numbers else None
        vals = [(self.primary_value(r), r) for r in rows]
        vals = [(v, r) for v, r in vals if v is not None]
        if not vals or (want is None and q.compare is None):
            return None
        ok = self._matching(q, rows, vals, want)
        if not ok and want is not None and q.compare is None and _STILL.search(q.raw):
            r = self._was_true_once(q, rows, want)
            # it was true once: say it moved, give today's figure, do not echo the old one
            if r is not None:
                return Result(
                    f"{lead}No, that is no longer the current figure. {self.line(r, q)}.",
                    self.sources([r], q),
                    True,
                    None,
                )
        if ok:
            shown = [r for _, r in ok][:6]
            return Result(
                f"{lead}Yes. " + "; ".join(self.line(r, q) for r in shown) + ".", self.sources(shown, q), True, None
            )
        if q.compare == "lt" or (q.compare is None and q.superlative == "low"):
            ex, tag = min(vals, key=lambda t: t[0])[1], "lowest"
        else:
            ex, tag = max(vals, key=lambda t: t[0])[1], "highest"
        if any(w in q.raw.lower() for w in ("still", "anymore", "any more", "no longer")) and len(vals) == 1:
            return Result(
                f"{lead}No, that is not the current figure. {self.line(ex, q)}.", self.sources([ex], q), True, None
            )
        return Result(f"{lead}No. The {tag} listed is {self.line(ex, q)}.", self.sources([ex], q), True, None)

    # ------------------------------------------------------------------ has it changed
    def _change(self, q, rows: list[Row]) -> Result | None:
        """'Has X changed?': the current value and the earlier value(s) from the key's merged history."""
        r = rows[0]
        attr = next(
            (
                a
                for a in (("apy",) if "apy" in q.attrs else ())
                + (*q.attrs, "apy", "apr", "rate", "dividend_rate", "fee")
                if a in r.vals
            ),
            None,
        )
        if attr is None:
            return None
        f = r.vals[attr]
        # only versions the answering snapshot could show
        hist = [h for h in self.fs.history_at(f.key, q.as_of) if h.fact.value is not None]
        if len(hist) < 2:
            return Result(
                f"{self.line(r, q)}. I see no change in this value across the website snapshots I have.",
                self.sources([r], q),
                True,
                None,
            )
        cur = hist[-1]
        # dated history, newest change first
        steps = [
            f"{display_value(a.fact)} from {month_day_year(a.valid_from)} to {month_day_year(b.valid_from)}"
            # earlier values named in a 'has it changed' answer
            for a, b in list(zip(hist, hist[1:]))[-3:]
        ]
        text = (
            f"{self.label(r)}: {display_value(cur.fact)} {attr_name(attr)} since {month_day_year(cur.valid_from)}; "
            f"previously {'; before that '.join(reversed(steps))}."
        )
        return Result(text, self.sources([r], q), True, None)

    # ------------------------------------------------------------------ sources / conflicts
    def sources(self, rows: list[Row], q) -> list[SourceRef]:
        out: list[SourceRef] = []
        seen = set()
        # rows that contribute sources
        for r in rows[:4]:
            f = sorted(r.vals.values(), key=rank_key, reverse=True)[0]
            same = sorted(
                (g for g in self.fs.view(q.as_of) if g.key == f.key and g.value_sig == f.value_sig),
                key=rank_key,
                reverse=True,
            )
            for g in [f] + same:
                if g.url in seen or len(out) >= _SOURCES_MAX:
                    continue
                # another page confirming the value must be at least a product page
                if g is not f and (authority(g) < 2.0 or url_fit(g.url, g.family) == 0):
                    continue
                seen.add(g.url)
                out.append(SourceRef(g.raw_url or g.url, g.snapshot_id, (g.row_text or "")[:200]))
        return out[:_SOURCES_MAX]

    def conflict_note(self, q, rows: list[Row]) -> str | None:
        """Which observation won and why, and what earlier snapshots showed."""
        f = sorted(rows[0].vals.values(), key=rank_key, reverse=True)[0]
        others = [
            g for g in self.fs.view(q.as_of) if g.key == f.key and g.value_sig != f.value_sig and g.confidence >= 0.6
        ]
        notes = []
        if others:
            o = sorted(others, key=rank_key, reverse=True)[0]
            why = (
                "a newer effective date"
                if (f.effective_date or "") > (o.effective_date or "")
                else "a more authoritative page type"
            )
            notes.append(
                f"Chose {display_value(f)} from {f.url} (effective {f.effective_date or 'n/a'}, {f.page_type}); "
                f"{o.url} shows {display_value(o)} (effective {o.effective_date or 'n/a'}, {o.page_type}), "
                f"which loses on {why}."
            )
        hist = [h for h in self.fs.history_at(f.key, q.as_of) if h.fact.value is not None]
        if len(hist) > 1 and not q.as_of:
            notes.append(f"Earlier snapshots showed {display_value(hist[-2].fact)} until {hist[-2].valid_until}.")
        return " ".join(notes) or None

_LIST_RATE_WORDS = re.compile(r"\b(rates?|apy|apr|dividends?|interest|yield|earn\w*|pay\w*|%)\b", re.I)


@dataclass
class FactAnswer:
    result: Result | None
    missing: list = field(default_factory=list)


class FactAnswerer:
    """Answers rate / fee / minimum / contact questions from `FactStore` (see module docstring)."""

    def __init__(self, kb, hub: HubFinder | None = None):
        self.kb = kb
        self.fs = kb.facts
        self.matcher = FactMatcher(kb)
        self.render = FactRenderer(kb)
        self.lists = FactLists(kb, self.matcher, hub)

    # ------------------------------------------------------------------ entry point
    def answer(self, q) -> FactAnswer:
        """A grounded `Result`, or None to let text retrieval (and its refusal rules) try."""
        found = self._answer_at(q)
        if (
            found.result is not None
            or q.as_of
            or q.kind not in ("rate", "fee", "min_open", "min_earn")
            or len(self.kb.views) < 2
        ):
            return found
        latest = self.kb.views[-1]
        for v in reversed(self.kb.views[:-1]):
            old = self._answer_at(replace(q, as_of=v.taken_on)).result
            if (
                old is not None
                and old.confident
                and old.sources
                and not any(normalize_url(sr.url) in latest.pages for sr in old.sources)
            ):
                return FactAnswer(self._as_last_published(old, v.taken_on))
        return found

    @staticmethod
    def _as_last_published(old: Result, taken_on: str) -> Result:
        lead = f"As of {month_day_year(taken_on)}, "
        body = old.text[len(lead) :] if old.text.startswith(lead) else old.text
        note = f" This was last published on {month_day_year(taken_on)}; that page is no longer on the website."
        return Result(body[:1].upper() + body[1:] + note, old.sources, True, old.conflict_note)

    # ------------------------------------------------------------------ one lookup
    def _answer_at(self, q) -> FactAnswer:
        self.matcher.ready()
        if q.kind == "contact":
            return FactAnswer(self.lists.contact(q))
        if q.intent == "url":
            return FactAnswer(self.lists.url_answer(q))
        text = self._product_wording(q)
        # a list question for these is not narrowed to the personal products
        segment = (
            None
            if (q.intent == "list" and not q.segment and q.family in ("credit_card", "certificate", "ira"))
            else "auto"
        )
        cands = self.fs.facts(text, as_of=q.as_of, segment=segment)
        if q.intent == "list" and q.kind == "rate" and not q.qualifiers and not _LIST_RATE_WORDS.search(q.raw):
            listed = self.lists.product_list(q)
            if listed is not None:
                return FactAnswer(listed)
        cands = self._with_paired_figures(q, text, segment, self._unhidden(q, text, segment, cands))
        pool = self.matcher.by_amount(q, self.matcher.widest_tiered_product(self.matcher.filter(q, cands), q))
        if not pool:
            return FactAnswer(None)
        unexplained = self.matcher.unexplained(q, pool)
        if unexplained and q.kind in ("rate", "min_open", "min_earn"):
            # the question names something the store does not have
            return FactAnswer(None, unexplained)
        rows = prune_rows(q, build_rows(self.matcher.pick_products(q, pool)))
        if not rows or (q.kind == "rate" and q.qualifiers.get("arm") and not any("arm" in r.quals for r in rows)):
            return FactAnswer(None)
        if unexplained and q.kind == "fee" and (len(rows) > 1 or len(unexplained) > 1):
            # nothing specific matched: a dump of every fee would not answer it
            return FactAnswer(None)
        res = self.render.render(q, rows, pool)
        if unexplained and q.kind == "fee" and res is not None:
            res = self._near_miss_fee(unexplained, rows, res)
        return FactAnswer(res)

    def _product_wording(self, q) -> str:
        """The product text the store is asked about."""
        text = product_text(q)
        for product, members_say in PRODUCT_ALIASES.items():
            if self.matcher.has_product(product) and re.search(members_say, text):
                # the product is named in the store, not guessed
                text = re.sub(members_say, product, text)
        return text

    def _unhidden(self, q, text: str, segment, cands: list) -> list:
        if cands:
            return cands
        known = self.matcher.known_words_only(text)
        if known and known != text and parse_query(known).qualifiers == parse_query(text).qualifiers:
            return self.fs.facts(known, as_of=q.as_of, segment=segment)
        return cands

    def _with_paired_figures(self, q, text: str, segment, cands: list) -> list:
        """The paired figure of each row (rate/APR, dividend rate/APY)."""
        if not cands or q.kind != "rate" or q.intent == "check":
            return cands
        have = {f.attribute for f in cands}
        for a in {PAIRED[w] for w in q.attrs if w in PAIRED} - have:
            cands = cands + self.fs.facts(text, attribute=a, as_of=q.as_of, segment=segment)
        return cands

    def _near_miss_fee(self, unexplained: list, rows: list, res: Result) -> Result:
        """Part of the question is not explained by any fee name: say so, show the nearest item."""
        return Result(
            f"I could not find a published fee for {' '.join(unexplained)}. The closest listed item is"
            f" {self.render.label(rows[0])}, which is a different fee. " + self.kb.voice.contact_phrase(),
            res.sources,
            False,
            res.conflict_note,
        )

