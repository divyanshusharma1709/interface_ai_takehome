"""Answerer: one question in, one `Result` out. The rules pipeline of one wording is an ordered list of resolvers, the
first one that has an answer wins; the Answerer wraps it with the refusals, the context split and the LLM cascade."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Protocol

from ..core.textutil import PHONE_ANY_SEP
from ..core.types import Result, SourceRef
from ..llm.parse import merge
from ..site_profile import INSTITUTION
from . import refuse
from .answer_type import genuine_rate_ask, wants_phone_line
from .branches import BranchTable
from .cascade import SLANG, Cascade, RuleOutcome
from .factqa import FactAnswerer
from .query import NUMERIC_KINDS, Query, clean_question, mentions_pronoun, parse, split_context
from .sentences import plain_text
from .textqa import TextAnswerer

FACT_KINDS = NUMERIC_KINDS + ("contact",)
_URGENT = re.compile(
    r"\b(?:lost|stolen|missing|misplaced|fraud\w*|unauthori[sz]ed|compromised)\b.*\b(?:card|account|debit|credit"
    r"|checkbook)\b|\b(?:card|debit|credit)\b.*\b(?:lost|stolen|missing|misplaced|fraud\w*|unauthori[sz]ed)\b",
    re.I,
)
_CARD_MISUSE = re.compile(r"\bcard\b.*\b(?:lost|stolen|unauthori[sz]ed|without your permission|fraud\w*)\b", re.I)
_MAIN_LINE_QUESTION = f"What phone number do I call to reach {INSTITUTION}?"


@dataclass
class Ask:
    """One wording as the resolvers see it, and the notes they leave for the cascade."""
    # parsed; with the LLM's fields merged in when `rules_only` is False
    q: Query
    clean: str
    # the as_of argument of the call (a date in the wording is `q.as_of`)
    as_of: str | None
    # the snapshot view the question is answered on
    view: object | None
    rules_only: bool = True
    # question words the fact store could not explain; None = the fact store was not asked
    missing: list | None = None
    # the answer is an extractive sentence of the text path (the reader may verify it)
    via_text: bool = False


class Resolver(Protocol):
    def try_answer(self, ask: Ask) -> Result | None: ...


class SnapshotLookups:
    """The per-snapshot lookups (branch table)."""

    def __init__(self):
        self._branches: dict = {}

    def branches(self, view) -> BranchTable:
        if view.snapshot_id not in self._branches:
            self._branches[view.snapshot_id] = BranchTable(view)
        return self._branches[view.snapshot_id]


_PHONE_EXT = re.compile(
    r"(?<![\d-])(?:\+?1[-. ]?)?\(?(\d{3})\)?[-. ]\s?(\d{3})[-. ](\d{4})(?!\d),?\s*ext(?:ension)?\.?\s*(\d{1,5})", re.I
)


def _card_line(view) -> Result | None:
    """The line a page states for a lost / stolen / misused card: its sentence, newest page first, sourced by every page with that line."""
    lines, seen = [], set()  # (number, ext, sentence, page), newest page first
    for page in sorted(view.pages.values(), key=lambda p: (p.publish_date or "", p.url), reverse=True):
        text = page.parsed.text or ""
        for m in _PHONE_EXT.finditer(text):
            s = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start()), text.rfind("? ", 0, m.start()))
            stop = re.compile(r"(?<![Ee]xt)(?<![Ee]xts)\. ").search(text, m.end())
            ends = [i for i in (stop.start() if stop else -1, text.find("\n", m.end())) if i >= 0]
            sent = plain_text(text[0 if s < 0 else s + 1 : min(ends) + 1 if ends else len(text)])[:320]
            # a sentence repeated on several pages is listed once (under the newest)
            key = ("".join(m.group(1, 2, 3)), m.group(4), sent[:80])
            if key not in seen:
                seen.add(key)
                lines.append((key[0], key[1], sent, page))
    best = next((x for x in lines if _CARD_MISUSE.search(x[2]) and len(x[2]) > 40), None)
    if best is None:
        return None
    digits, ext, sent, _ = best
    text = sent if digits in re.sub(r"\D", "", sent) else f"{sent} The number is {digits[:3]}-{digits[3:6]}-{digits[6:]} ext. {ext}."
    same = sorted((x for x in lines if x[:2] == best[:2]), key=lambda x: x[3].page_type in ("newsroom", "blog"))
    refs, urls = [], set()
    for _, _, snippet, page in [best] + same:
        if page.raw_url not in urls and len(refs) < 5:
            urls.add(page.raw_url)
            refs.append(SourceRef(page.raw_url, page.snapshot_id, snippet[:200]))
    return Result(text, refs, True, None)


class UrgentLine:
    """A lost card is an emergency: the line stated for exactly this (card services), else the main line to call."""

    def __init__(self, lookups: SnapshotLookups, facts: FactAnswerer, parse):
        self.lookups, self.facts, self.parse = lookups, facts, parse

    def try_answer(self, ask: Ask) -> Result | None:
        q = ask.q
        if not (
            _URGENT.search(q.raw)
            and q.kind not in NUMERIC_KINDS
            and q.intent in ("howto", "value", "info", "contact")
            and not PHONE_ANY_SEP.search(q.raw)
        ):
            return None
        res = _card_line(ask.view) if ask.view is not None and wants_phone_line(q.raw.lower()) else None
        if res is None:
            res = self.facts.answer(self.parse(_MAIN_LINE_QUESTION, ask.as_of)).result
        return res if res is not None and res.confident else None


class BranchLookup:
    def __init__(self, kb, lookups: SnapshotLookups):
        self.kb, self.lookups = kb, lookups

    def try_answer(self, ask: Ask) -> Result | None:
        q, view = ask.q, ask.view
        if view is None or q.kind in NUMERIC_KINDS:
            return None
        ref = str(q.as_of)[:10] if q.as_of else view.taken_on
        return self.lookups.branches(view).answer(q, ref, q.as_of is not None, self.kb.voice)


class FactLookup:
    def __init__(self, text: TextAnswerer, facts: FactAnswerer):
        self.text, self.facts = text, facts

    def try_answer(self, ask: Ask) -> Result | None:
        q = ask.q
        about_a_place = q.kind == "contact" and bool(self.text.entities.of(q))
        if (
            q.kind in FACT_KINDS
            and not about_a_place
            and q.intent not in ("branch", "howto")
            and (q.intent != "url" or q.kind == "rate")
        ):
            found = self.facts.answer(q)
            ask.missing = found.missing
            return found.result
        return None


class TextLookup:

    def __init__(self, kb, text: TextAnswerer):
        self.kb, self.text = kb, text

    def try_answer(self, ask: Ask) -> Result:
        q = ask.q
        res = self.text.answer(q, strict_numeric=genuine_rate_ask(q))
        ask.via_text = ask.rules_only and res.extractive
        why = refuse.unpublished_variant(q, res, ask.missing or [])
        return self.kb.voice.refusal(why) if why else res


class RuleChain:
    """Runs one wording through the resolvers, remembering the outcome of each (wording, as_of, LLM fields)."""

    def __init__(self, kb, parse, resolvers: list[Resolver]):
        self.kb, self.parse, self.resolvers = kb, parse, resolvers
        self._outcomes: dict[tuple, RuleOutcome] = {}

    def run(self, clean: str, as_of, v: dict | None = None) -> RuleOutcome:
        key = (clean, str(as_of), len(self.kb.views), json.dumps(v, sort_keys=True) if v else "")
        if key not in self._outcomes:
            self._outcomes[key] = self._resolve(clean, as_of, v)
        return self._outcomes[key]

    def _resolve(self, clean: str, as_of, v: dict | None) -> RuleOutcome:
        # rules first; valid LLM fields override (as_of argument still wins)
        q = self._anchored(merge(self.parse(clean, as_of), v), as_of)
        ask = Ask(q, clean, as_of, self.kb.view_for(q.as_of), rules_only=v is None)
        for resolver in self.resolvers:
            res = resolver.try_answer(ask)
            if res is not None:
                return RuleOutcome(res, ask.via_text, None if ask.missing is None else tuple(ask.missing))
        raise AssertionError("the last resolver always answers")

    def _anchored(self, q: Query, as_of) -> Query:
        views = self.kb.views
        if (
            q.as_of
            and as_of in (None, "")
            and q.kind == "text"
            and q.intent not in ("value", "check", "change")
            and views
            and q.as_of[:10] < views[0].taken_on
        ):
            return replace(
                q, year_hint=int(q.as_of[:4]), month_hint=int(q.as_of[5:7]) if len(q.as_of) >= 7 else None, as_of=None
            )
        return q


_SHE_HE = re.compile(r"\b(?:she|he)\b", re.I)


def _policy_knob(name: str) -> property:
    """An attribute of the reader policy, also settable on the Answerer (tools and tests tune them per instance)."""
    return property(
        lambda self: getattr(self.cascade.policy, name), lambda self, value: setattr(self.cascade.policy, name, value)
    )


def _cascade_part(name: str) -> property:
    return property(lambda self: getattr(self.cascade, name), lambda self, value: setattr(self.cascade, name, value))


class Answerer:
    """Deterministic router with a per-(question, as_of) cache."""

    _SLANG = SLANG
    VERIFY, WEAK_COV = _policy_knob("verify"), _policy_knob("weak_cov")
    VERIFY_COV, VERIFY_COV_TYPED = _policy_knob("verify_cov"), _policy_knob("verify_cov_typed")
    llm, reader, budget = _cascade_part("llm"), _cascade_part("reader"), _cascade_part("budget")

    def __init__(self, kb):
        self.kb = kb
        self.text = TextAnswerer(kb)
        self.facts = FactAnswerer(kb, hub=self.text.hubs.of_question)
        lookups = SnapshotLookups()
        self._chain = RuleChain(
            kb,
            self.parse,
            [
                UrgentLine(lookups, self.facts, self.parse),
                BranchLookup(kb, lookups),
                FactLookup(self.text, self.facts),
                TextLookup(kb, self.text),
            ],
        )
        # (looked up per call: tests swap `_answer_with`)
        self.cascade = Cascade(kb, self.text, self.parse, lambda clean, as_of, v: self._answer_with(clean, as_of, v))
        self._answers: dict[tuple, Result] = {}

    def latest(self) -> str:
        return self.kb.views[-1].taken_on if self.kb.views else ""

    def parse(self, question: str, as_of=None) -> Query:
        return parse(question, as_of, self.latest())

    def answer(self, question: str, as_of=None) -> Result:
        """Rules first."""
        self.cascade.begin()
        key = (question, str(as_of), len(self.kb.views))
        if key not in self._answers:
            clean = clean_question(question)
            self._answers[key] = (self.cascade.answer(question, clean, as_of) if self.cascade.live
                                  else self._answer_with(clean, as_of, None).result)
        return self._answers[key]

    def reader_info(self, question: str, as_of=None) -> dict | None:
        return self.cascade.reader_info(question, as_of)

    def _answer_with(self, clean: str, as_of, v: dict | None) -> RuleOutcome:
        """The rules' outcome for a cleaned question (`v` = fields the LLM parser supplied, or None)."""
        context, ask = split_context(clean)
        why = None if (v and v.get("answerable") == "yes") else refuse.out_of_scope(ask if context else clean)
        if why:
            return RuleOutcome(self.kb.voice.refusal(why))
        out = self._chain.run(clean, as_of, v)
        beyond = refuse.beyond_the_data(clean, self.latest(), lambda: self.parse(clean, as_of).kind, out.result)
        if beyond:
            out = replace(out, result=self.kb.voice.refusal(beyond))
        # a declarative lead-in ('I live in Dubuque.', 'My daughter is 16.') is context: when the whole wording finds
        # nothing, the question alone may
        if context and not out.result.confident and not mentions_pronoun(ask):
            alone = self._chain.run(_SHE_HE.sub("a member", ask), as_of)
            if alone.result.confident:
                out = replace(
                    out, result=alone.result, missing=alone.missing if alone.missing is not None else out.missing
                )
        return out
