"""The passage reader: the passages shown to the model, its request and `LLMReader`."""

from __future__ import annotations

import hashlib
import json
import re

from ..core.types import Result
from ..qa.textqa import render
from .client import SOLUTION_DIR, LLMSettings, Session, normalize_question
from .grounding import ground, same_evidence
from .prompts import READER_SYSTEM, READER_TOOL, TYPE_LABEL

# ---- passages
PASSAGE_CHARS = 900
PASSAGE_LEAD_CHARS = 250  # context kept before the matched sentence
OVERFLOW_SLACK = 500
STALE_TYPES = ("blog", "newsroom")


def _plain(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", text)).strip()


def _passage_text(c) -> str:
    body = render(c.chunk.text)
    i = body.find(c.sent[:40])
    lo = max(0, (i if i >= 0 else 0) - PASSAGE_LEAD_CHARS)
    span = len(body) - lo if len(body) - lo <= PASSAGE_CHARS + OVERFLOW_SLACK else PASSAGE_CHARS
    txt = _plain(body[lo : lo + span])
    head = (c.chunk.heading_path or "").strip()
    if head and head.lower() not in txt.lower()[:200]:
        txt = f"{head}: {txt}"
    return txt


def build_passages(text_answerer, q, view, n: int = 5) -> tuple[list[dict], float]:
    """Top passages for the reader: [{n, url, snapshot, type, date, stale, label, text}]."""
    cands, cov = text_answerer.reading_set(q, n)
    taken = view.taken_on if view is not None else ""
    out = []
    for c in cands:
        page = c.page
        date = page.publish_date or page.effective_date or ""
        old = bool(
            date and taken and date[:4].isdigit() and taken[:4].isdigit() and int(taken[:4]) - int(date[:4]) >= 1
        )
        stale = page.page_type in STALE_TYPES
        label = TYPE_LABEL.get(page.page_type, "website page")
        note = ("dated article" + (", more than a year old" if old else "")) if stale else "current official page"
        out.append(
            {
                "n": len(out) + 1,
                "url": page.raw_url,
                "snapshot": page.snapshot_id,
                "type": page.page_type,
                "date": date,
                "stale": stale,
                "old": old,
                "label": f"{label}, {note}",
                "text": _passage_text(c),
                "title": page.title or "",
            }
        )
    # official pages ahead of dated articles when their retrieval scores were close: stable sort keeps the retrieval
    # order inside each group
    out.sort(key=lambda p: (p["stale"] and p["old"],))
    for i, p in enumerate(out, 1):
        p["n"] = i
    return out, cov


def passages_digest(passages: list[dict], candidate: str = "") -> str:
    return hashlib.sha256(json.dumps([[p["url"], p["text"]] for p in passages] + [candidate]).encode()).hexdigest()[:16]


def reader_user(question: str, as_of_date: str, passages: list[dict], candidate: str = "") -> str:
    lines = [f"Question: {question}", f"The website as of: {as_of_date}", "", "Passages (best match first):"]
    for p in passages:
        lines.append(
            f"\n[{p['n']}] URL: {p['url']}\nType: {p['label']}"
            + (f"; dated {p['date']}" if p["date"] else "")
            + f"\n{p['text']}"
        )
    if candidate:
        lines.append(f"\nCandidate answer (verify it): {candidate}")
    return "\n".join(lines)


# ---- reader
READER_VERSION = "r2"
READ_TIMEOUT = 2.8  # seconds for one reader call
MAX_TOKENS = 500
CANDIDATE_CHARS = 600
MAX_QUOTES = 5


def request_text(question: str, passages: list[dict], candidate: str) -> str:
    """The text a reader request is cached, memoised and logged under."""
    return f"{question}\x1f{passages_digest(passages, candidate)}"


class LLMReader(Session):
    """The reader's side of a session: prompt, tool, cache key and reply validation."""

    version = READER_VERSION
    max_tokens = MAX_TOKENS
    system = READER_SYSTEM
    tool = READER_TOOL

    def __init__(self, latest: str, settings: LLMSettings, timeout: float | None = None):
        super().__init__(settings, float(timeout if timeout is not None else READ_TIMEOUT))
        self.latest = latest

    def cache_key(self, text: str, as_of, model: str) -> str:
        return hashlib.sha256(
            json.dumps([self.version, model, normalize_question(text), str(as_of or "")]).encode()
        ).hexdigest()[:32]

    def user(self, text: str, payload) -> str:
        question, view_date, passages, candidate = payload
        return reader_user(question, view_date or self.latest, passages, candidate)

    def validate(self, raw):
        if not isinstance(raw, dict) or not ("better_answer_text" in raw or "answer_text" in raw):
            return None
        ans = str(raw.get("better_answer_text", raw.get("answer_text")) or "")
        verdict = str(raw.get("verdict") or "").lower()
        if verdict not in ("answers", "does_not_answer"):
            verdict = "does_not_answer"
        yn = str(raw.get("answerable") or ("yes" if ans else "no")).lower()
        if yn not in ("yes", "no"):
            return None
        return {
            "verdict": verdict,
            "answerable": yn,
            "answer_text": ans,
            "quotes": [str(x) for x in (raw.get("quotes") or []) if isinstance(x, str)][:MAX_QUOTES],
            "source_urls": [str(x) for x in (raw.get("source_urls") or []) if isinstance(x, str)][:MAX_QUOTES],
        }

    def read(
        self, question: str, as_of, passages: list[dict], view_date: str = "", candidate: str = ""
    ) -> tuple[Result | None, str]:
        """One call: rescue (no candidate) or verify (candidate given)."""
        if self.mode == "off" or not passages:
            return None, "off"
        candidate = (candidate or "")[:CANDIDATE_CHARS]
        out = self.get(
            request_text(question, passages, candidate),
            as_of,
            payload=(question, view_date, passages, candidate),
            fixture_text=question,
        )
        if out is None:
            return None, "no_output"
        if candidate and out["verdict"] == "answers":
            return None, "confirmed"
        if out["answerable"] != "yes" or not out["answer_text"]:
            return None, "none_answer"
        res, why = ground(out, passages, question)
        return res, ("used" if res is not None else f"rejected:{why}")

    def info_for_read(self, question: str, as_of, passages: list[dict], candidate: str = "") -> dict:
        return self.info_for(request_text(question, passages, (candidate or "")[:CANDIDATE_CHARS]), as_of)


def make_reader(kb, folder=None, timeout: float | None = None, **settings) -> LLMReader:
    """The reader for a knowledge base; inactive unless KS_LLM / an API key turns it on."""
    resolved = LLMSettings.resolve(folder=folder or SOLUTION_DIR, **settings)
    return LLMReader(kb.views[-1].taken_on if kb.views and resolved.active else "", resolved, timeout)


__all__ = ["LLMReader", "build_passages", "make_reader", "same_evidence"]
