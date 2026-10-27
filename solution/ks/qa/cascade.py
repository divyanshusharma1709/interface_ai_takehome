"""The optional LLM cascade over the rules' answer (inactive, and free, unless KS_LLM / an API key turns it on)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from functools import cached_property

from ..core.tokens import tokenize as tokenize_text
from ..core.types import Result
from ..llm.parse import HARD_NO, Budget, make_parser, merge, supported_fields
from ..llm.read import build_passages, make_reader, same_evidence
from ..site_profile import INSTITUTION
from .answer_type import AMOUNT_ASK, RATE_ASK, expected_detail
from .query import NUMERIC_KINDS, clean_question, split_context
from .refuse import RESCUABLE


# words of a chat question that make its parse unsure
SLANG = frozenset(
    "u ur r rn lol pls plz thx ty wanna gonna gimme lemme ya idk btw tho cuz bc ppl gotta tryna kinda".split()
)
# shorter words are not checked for misspelling
_MIN_WORD = 5
# words that do not say which fee
_FEE_NOISE = {"fee", "fees", "charge", "charges", "cost", "price", "much", INSTITUTION.lower()}
_STEM = 5
_QUESTION_WORDS = re.compile(r"\b(?:where|when|who|whom|whose|why|how|which|what|can|does|is|are|do|will)\b")


@dataclass(frozen=True)
class RuleOutcome:
    """What the rules made of a wording: the answer, and what the LLM cascade needs to know about how they got it."""
    result: Result
    via_text: bool = False
    missing: tuple | None = None


@dataclass
class ReadPolicy:
    """When the reader is asked (tests and tools change these)."""
    # check confident text-path answers with the reader (False: only rescue refusals)
    verify: bool = True
    # Reader settings
    # a confident extractive answer covering less of the question than this is checked by the reader
    weak_cov: float = 0.5
    # no typed detail asked: an answer covering at least this share of the question's content words is not checked
    verify_cov: float = 0.88
    # a typed detail (hours, phone, amount, count, date, URL) is present in the answer: the same threshold
    verify_cov_typed: float = 0.6


class Cascade:
    """The reader and parser second opinions over the rules pass `rules(clean, as_of, v)`."""

    def __init__(self, kb, text, parse, rules):
        self.kb, self.text, self.parse, self.rules = kb, text, parse, rules
        # the parser
        self.llm = make_parser(kb)
        # rescues text-path answers by reading the retrieved passages
        self.reader = make_reader(kb)
        # KS_LLM_BUDGET: wall time for all LLM calls within one answer()
        self.budget = Budget()
        self.policy = ReadPolicy()
        # canonical query -> final answer
        self._final: dict = {}
        # canonical query -> the first phrasing seen: one reader call per meaning
        self._canon_rep: dict = {}
        # canonical query -> its passages and their coverage
        self._canon_pass: dict = {}
        # question -> what the reader was sent (for `reader_info`)
        self._read_seen: dict = {}

    # ------------------------------------------------------------------ entry points
    def begin(self) -> None:
        """Each answer rebinds the budget (tests swap the clients) and starts its clock."""
        self.llm.budget = self.reader.budget = self.budget
        self.budget.start()

    @property
    def live(self) -> bool:
        return bool(self.llm.active or self.reader.active)

    def answer(self, question: str, clean: str, as_of) -> Result:
        """The final answer of a question; a paraphrase of one already answered gets the same."""
        canon = self.canon(clean, as_of)
        if canon in self._final:
            return self._final[canon]
        outcome = self.rules(clean, as_of, None)
        self._canon_rep.setdefault(canon, question)
        res = self._escalate(question, clean, as_of, outcome)
        self._final[canon] = res
        return res

    def canon(self, clean: str, as_of) -> tuple:
        q = self.parse(clean, as_of)
        # product names live in the words: never merge two different products
        words = tuple(sorted(set(tokenize_text(q.raw)) | set(_QUESTION_WORDS.findall(q.raw.lower()))))
        return (
            q.intent,
            q.kind,
            tuple(sorted(q.attrs or ())),
            q.family,
            q.segment,
            tuple(sorted((str(k), str(v)) for k, v in (q.qualifiers or {}).items())),
            str(q.as_of),
            q.year_hint,
            q.month_hint,
            tuple(sorted(str(n) for n in (q.numbers or ()))),
            words,
            str(as_of or ""),
        )

    def reader_info(self, question: str, as_of=None) -> dict | None:
        seen = self._read_seen.get((" ".join(question.lower().split()), str(as_of or "")))
        if not seen:
            return None
        rep, passages, cand = seen
        info = dict(self.reader.info_for_read(rep, as_of, passages, cand))
        usage = info.pop("usage", None) or {}
        if "output_tokens" in usage:
            info["output_tokens"] = usage["output_tokens"]
        return info

    def _escalate(self, question: str, clean: str, as_of, outcome: RuleOutcome) -> Result:
        if self.reader.active:
            done, res = self._read(question, clean, as_of, outcome)
            if done:
                return res
        if not self.llm.active:
            return outcome.result
        return self._second_opinion(question, clean, as_of, outcome)

    # ------------------------------------------------------------------ the reader
    def _leaves_to_rules(self, q, outcome: RuleOutcome) -> bool:
        """Answers the reader is never asked about: deterministic ones, and refusals that are a firm no."""
        rules = outcome.result
        if rules.confident:
            if not outcome.via_text:
                # a fact / table answer stays deterministic
                return True
            return not self.policy.verify and not (
                q.kind == "text" or (q.kind == "contact" and bool(self.text.entities.of(q)))
            )
        if not RESCUABLE.search(rules.conflict_note or ""):
            # advice / off-topic / personal data / unknown name / missing variant: a firm no
            return True
        # 'rate on a 13-month certificate' with no such row: the refusal is the answer, not a reading problem
        return (
            q.kind in NUMERIC_KINDS
            and bool(RATE_ASK.search(q.raw))
            and not AMOUNT_ASK.search(q.raw)
            and " and " not in q.raw.lower()
        )

    def _passages(self, canon: tuple, rep: str, question: str, clean: str, as_of, q, view) -> tuple[list, float]:
        """The passages the reader sees; canonical-equal phrasings share passages (and the reader call)."""
        if canon not in self._canon_pass:
            rq = q if rep == question else self.parse(clean_question(rep), as_of)
            self._canon_pass[canon] = build_passages(self.text, rq, view)
        return self._canon_pass[canon]

    def _read(self, question: str, clean: str, as_of, outcome: RuleOutcome):
        """(done, result). done=False -> go on to the parser cascade."""
        rules = outcome.result
        q = self.parse(clean, as_of)
        view = self.kb.view_for(q.as_of)
        if self._leaves_to_rules(q, outcome) or view is None:
            return False, rules
        canon = self.canon(clean, as_of)
        rep = self._canon_rep.get(canon, question)
        passages, cov = self._passages(canon, rep, question, clean, as_of, q, view)
        if not passages:
            return False, rules
        verify = rules.confident and self.policy.verify
        if rules.confident and not self.policy.verify and cov >= self.policy.weak_cov:
            return False, rules
        if verify and self.strong(q, rules, passages, cov):
            self.llm.set_outcome(question, as_of, "skipped:strong_extractive")
            # strong answers are not worth a model call (latency)
            return False, rules
        cand = rules.text if verify else ""
        self._read_seen[(" ".join(question.lower().split()), str(as_of or ""))] = (rep, passages, cand)
        res, status = self.reader.read(rep, as_of, passages, view.taken_on, cand)
        return self._reader_verdict(question, as_of, rules, res, status)

    def _reader_verdict(self, question: str, as_of, rules: Result, res, status: str):
        """(done, result) after the reader's reply."""
        outcome = lambda tag: self.llm.set_outcome(question, as_of, tag)
        if status == "no_output" and self.reader.mode in ("live", "record"):
            outcome("kept_rules:reader_failed" + (":budget" if self.reader.budget_skipped else ""))
            # a slow or failed call is not followed by a second call (latency): the rules answer stands
            return True, rules
        if status in ("off", "no_output"):
            # not recorded (replay) / not applicable: the parser cascade (or the rules) decides
            return False, rules
        if res is not None and rules.confident:
            why = same_evidence(rules.text, res, rules.sources)
            # the reader re-worded the same evidence: the extractive sentences keep their detail
            if why:
                outcome(f"verified:{why}")
                return True, rules
        if res is not None:
            outcome("used:reader_verify" if rules.confident else "used:reader")
            return True, res
        if status == "confirmed":
            outcome("verified:kept_rules")
            return True, rules
        if status == "none_answer":
            # the reader says the passages do not answer it either: a refusal beats a nearby wrong sentence
            if rules.confident:
                outcome("used:reader_refuse")
                return True, self.kb.voice.refusal(
                    "the reader found that the retrieved passages do not answer the question"
                )
            outcome("kept_rules:reader_answerable_no")
            return True, rules
        outcome(f"kept_rules:reader_{status.split(':', 1)[-1]}")
        return True, rules

    def strong(self, q, rules: Result, passages: list, cov: float) -> bool:
        url = rules.sources[0].url.rstrip("/") if rules.sources else ""
        src = next((p for p in passages if p["url"].rstrip("/") == url), None)
        if src is None or (src["stale"] and src["old"]):
            return False
        need = expected_detail(q)
        if need is not None:
            return bool(need.search(re.sub(r"https?://\S+", "", rules.text))) and cov >= self.policy.verify_cov_typed
        if q.intent == "url" and "http" in rules.text:
            return cov >= self.policy.verify_cov_typed
        return cov >= self.policy.verify_cov and q.intent != "list"

    # ------------------------------------------------------------------ the parser
    def _second_opinion(self, question: str, clean: str, as_of, outcome: RuleOutcome) -> Result:
        llm, rules = self.llm, outcome.result
        why = None if not rules.confident else self.uncertain(clean, as_of, question, outcome.missing)
        trigger = "rescue" if not rules.confident else ("uncertain_parse" if why else None)
        if trigger is None:
            llm.set_outcome(question, as_of, "skipped:confident_rules")
            return rules
        v = llm.get(question, as_of)
        if not v:
            llm.set_outcome(question, as_of, "kept_rules:no_llm_output" + (":budget" if llm.budget_skipped else ""))
            return rules
        # advice / off-topic / someone's own account: a firm no.
        if trigger == "rescue" and v.get("answerable") == "no" and v.get("reason") in HARD_NO:
            # (Never used to overturn a confident rule answer: the model mislabels real questions off_topic often enough
            # to cost more than it saves.)
            llm.set_outcome(question, as_of, f"used:{trigger}")
            return self.kb.voice.refusal(
                f"the question is not answerable from the public website ({v['reason'].replace('_', ' ')})"
            )
        dropped = False
        # only fields the question's own words support may change a confident rule answer
        if trigger == "uncertain_parse":
            supported = supported_fields(v, self.parse(clean, as_of), clean, llm.cat)
            dropped, v = supported != v, supported
        kept = self._fee_item_mismatch(clean, v, merge(self.parse(clean, as_of), v), rules)
        if kept:
            llm.set_outcome(question, as_of, kept)
            return rules
        alt = self.rules(clean, as_of, v).result
        kept = self._llm_adds_nothing(trigger, clean, as_of, v, rules, alt, dropped)
        llm.set_outcome(question, as_of, kept or f"used:{trigger}")
        return rules if kept else alt

    def _fee_item_mismatch(self, clean: str, v: dict, moved, rules: Result) -> str | None:
        """The outcome tag when the fee row the model picked is not the fee the question asked about."""
        if moved.kind != "fee" or not v.get("product"):
            return None
        asked = {x[:_STEM] for x in tokenize_text(clean) if x not in _FEE_NOISE}
        row = {x[:_STEM] for x in tokenize_text(v["product"]) if x not in _FEE_NOISE}
        # a fee the question never names is a nearby different fee, not the one asked about
        if row and not (asked & row):
            return "kept_rules:fee_item_unmatched"
        if (
            not rules.confident
            and "closest listed item is" in rules.text
            and v["product"].lower() in rules.text.lower()
        ):
            # the rules already weighed this very row and found it a different fee
            return "kept_rules:fee_item_is_the_nearest_miss"
        return None

    def _llm_adds_nothing(
        self, trigger: str, clean: str, as_of, v: dict, rules: Result, alt: Result, dropped: bool
    ) -> str | None:
        """The outcome tag when the LLM-assisted answer should not replace the rules' answer, else None."""
        if not (alt.confident and alt.sources):
            return "kept_rules:llm_not_confident"
        if trigger != "uncertain_parse":
            return None
        base = self.parse(clean, as_of)
        moved = merge(base, v)
        fields = ("kind", "attrs", "family", "segment", "qualifiers", "as_of", "product")
        if all(getattr(base, f) == getattr(moved, f) for f in fields) or alt.text == rules.text:
            return "kept_rules:unsupported_fields" if dropped else "kept_rules:llm_adds_nothing"
        if rules.confident and rules.text.strip().rstrip(".") in alt.text and len(alt.text) > len(rules.text):
            # the model only widened a precise answer into a list that contains it: the specific one stands
            return "kept_rules:llm_broader"
        return None

    # ------------------------------------------------------------------ parse-uncertainty signals: few and generic
    def uncertain(self, clean: str, as_of, raw: str, missing) -> str | None:
        """Why the rule parse of `clean` looks unsure (a reason), or None."""
        if any(w in SLANG for w in re.findall(r"[a-z0-9']+", (raw or clean).lower())):
            return "slang"
        context, _ = split_context(clean)
        if context:
            return "personal_context"
        q = self.parse(clean, as_of)
        if q.kind in NUMERIC_KINDS:
            if not q.family and q.kind != "fee":
                return "no_product"
            if missing:
                return "unexplained_words"
            if self._unplaced_number(q, clean.lower()):
                return "unplaced_number"
        return self._misspelled(re.findall(r"[a-z0-9']+", clean.lower()))

    @staticmethod
    def _unplaced_number(q, low: str) -> bool:
        """A figure in the wording that the parse did not place in any field."""
        tm = q.qualifiers.get("term_months")
        vals = [tm, q.qualifiers.get("amount"), q.qualifiers.get("model_year"), *q.numbers]
        placed = {str(int(x)) for x in vals if isinstance(x, (int, float)) and float(x) == int(x)}
        if isinstance(tm, (int, float)):
            placed.add(str(int(tm // 12)))
        for m in re.finditer(r"(?<![\w$.])(\d[\d,]*)(k?)(?![\w%.])", low):
            num = m.group(1).replace(",", "")
            if (
                num not in placed
                and not re.fullmatch(r"(?:19|20)\d\d", num)
                # a bare 1-3 in a question ('2 people') is not a figure the parse must place
                and not (m.group(2) == "" and int(num or 0) in (1, 2, 3))
            ):
                return True
        return False

    @cached_property
    def _site_words(self) -> list[str]:
        view = self.kb.views[-1]
        return [w for w in view.index.idf if w.isalpha() and len(w) >= _MIN_WORD]

    def _misspelled(self, words) -> str | None:
        """'misspelling' when a word the site never uses is a near miss of one it does."""
        if not self.kb.views:
            return None
        idf = self.kb.views[-1].index.idf
        for w in words:
            if len(w) < _MIN_WORD or not w.isalpha():
                continue
            toks = tokenize_text(w)
            if not toks or toks[0] in idf or w in idf:
                continue
            pool = [x for x in self._site_words if x[0] == toks[0][0] and abs(len(x) - len(toks[0])) <= 2]
            # similarity to a known site word that makes an unknown word a misspelling
            if difflib.get_close_matches(toks[0], pool, n=1, cutoff=0.84):
                return "misspelling"
        return None
