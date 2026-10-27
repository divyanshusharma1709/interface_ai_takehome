"""Answer verification for the reader: nothing the model wrote is trusted until `ground` has checked it against the
passages."""

from __future__ import annotations

import re

from ..core.textutil import NUMBER_TENS, NUMBER_UNITS, SENT_BREAK, strip_md_links
from ..core.tokens import tokenize
from ..core.types import Result, SourceRef
from ..site_profile import ACRONYMS_SITE, COMMON_CAPS_SITE, PLACE_WORDS_SITE, SOFT_WORDS_SITE

MAX_ANSWER_CHARS = 600
MAX_ANSWER_SENTENCES = 4
MIN_QUOTE_CHARS = 8
MAX_SOURCES = 3
SOURCE_QUOTE_CHARS = 200
MAX_DAY_OF_MONTH = 31

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "−": "-", " ": " "})
_NUMWORD = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "first": 1,
    "second": 2,
    "third": 3,
    "half": 0.5,
}
_NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
_YEAR = re.compile(r"(?:19|20)\d\d")
# capitalised words that begin sentences or are filler in answers: never evidence of an invented name
_words = lambda text: set(text.split())
_SOFT = (
    _words(
        "the a an yes no it this that these those you your there members member as for to on in at if please "
        "i we our they each all both any some when with by from or and but per up after before see call visit "
        "contact once also currently most new not only can may will must however"
    )
    | SOFT_WORDS_SITE
)
_ACRO = (
    _words(
        "apr apy ira heloc atm atms cd cds pin visa ach ncua fdic ssn id fico cu cus us usa irs llc ceo cfo "
        "coo cmo cio avp svp vp hsa roth rv ev ssa faq pdf url am pm nsf hr ira's apys aprs ltv dti pmi arm "
        "fha va usda ucc"
    )
    | ACRONYMS_SITE
)
_COMMON_CAP = (
    _words(
        "annual health low high savings checking money market credit union card cards loan loans account "
        "accounts rewards reward mobile online digital banking business personal home auto vehicle student "
        "mortgage member members membership certificate certificates share draft free standard premium "
        "dividend dividends interest rate rates fee fees overdraft privilege balance minimum maximum monthly "
        "daily weekly yearly early education retirement insurance investment investments financial wealth "
        "planning services service center branch branches lobby drive video teller live holiday club "
        "community fixed variable term terms new used first second third also since its she he her his they "
        "them their we our us you your it this that these those there here what when where which who why how "
        "if as at by for from in of on or to with monday tuesday wednesday thursday friday saturday sunday "
        "january february march april may june july august september october november december up down over "
        "under about after before during until through per each every all any some most more less many much "
        "one two three four five six seven eight nine ten yes no not only just then than so but and nor yet "
        "while because although however therefore thus note please see call visit contact apply open close"
    )
    | COMMON_CAPS_SITE
)
_DAYS_MONTHS = (
    _words(
        "monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
        "august september october november december"
    )
    | PLACE_WORDS_SITE
)
_REFUSAL = re.compile(
    r"\b(?:i don't know|i do not know|not (?:stated|mentioned|specified|published|provided)|does(?: not|n't) (?:say|"
    r"state|mention|specify)|cannot (?:be )?(?:determine|find)|no information)\b",
    re.I,
)

_UNITS = {"zero": 0, **NUMBER_UNITS}
_SPELLED = re.compile(
    r"\b(?:(?:%s)(?:[\s-]+(?:%s))?|(?:%s))\b"
    % ("|".join(NUMBER_TENS), "|".join(w for w in _UNITS if 0 < _UNITS[w] < 10), "|".join(_UNITS)),
    re.I,
)


# ---------------------------------------------------------------------------------------------------- text and number
# helpers
def norm_text(s: str) -> str:
    s = (s or "").translate(_QUOTES)
    s = re.sub(r"[*_`#>]|\\", "", s)
    s = strip_md_links(s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _quote_in(nq: str, passage: str) -> bool:
    if nq in passage:
        return True
    squash = lambda t: re.sub(r"[^a-z0-9.,$%:/]+", "", t)
    flat = squash(passage)
    pieces = [x for x in re.split(r"\.{3}|…", nq) if len(squash(x)) >= 6]
    if not pieces:
        return False
    at = 0
    for piece in pieces:
        i = flat.find(squash(piece), at)
        if i < 0:
            return False
        at = i + len(squash(piece))
    return True


def _canon(tok: str) -> str:
    t = tok.replace("$", "").replace(",", "").replace("%", "")
    try:
        f = float(t)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return t


def _numbers(text: str) -> set:
    return {_canon(m.group(0)) for m in _NUM.finditer(text.translate(_QUOTES))}


def _spelled(text: str) -> set:
    out = {str(v if v != int(v) else int(v)) for w, v in _NUMWORD.items() if re.search(rf"\b{w}\b", text.lower())}
    for m in _SPELLED.finditer(text):
        parts = re.split(r"[\s-]+", m.group(0).lower())
        out.add(str(sum(NUMBER_TENS.get(x, _UNITS.get(x, 0)) for x in parts)))
    return out


def _is_year(num: str) -> bool:
    return bool(_YEAR.fullmatch(num))


def _years(text: str) -> set:
    return {n for n in _numbers(text) if _is_year(n)}


# ---------------------------------------------------------------------------------------------------- the checks
# (each: a reason, or None)
def _reply_fields(out) -> tuple | str:
    if not isinstance(out, dict):
        return "no_output"
    if str(out.get("answerable", "")).lower() != "yes":
        return "answerable_no"
    ans = str(out.get("answer_text") or "").strip()
    quotes = [str(x).strip() for x in (out.get("quotes") or []) if str(x).strip()]
    urls = [str(u).strip() for u in (out.get("source_urls") or [])]
    if not ans or not quotes:
        return "empty"
    if len(ans) > MAX_ANSWER_CHARS or len(re.findall(r"[.!?](?:\s|$)", ans)) > MAX_ANSWER_SENTENCES:
        return "too_long"
    if _REFUSAL.search(ans):
        return "answer_is_refusal"
    return ans, quotes, urls


def _quote_homes(quotes: list, cited: list, norm_pass: dict) -> list | str:
    used: list[dict] = []
    for qt in quotes:
        nq = norm_text(qt)
        if len(nq) < MIN_QUOTE_CHARS:
            return "quote_too_short"
        home = next((p for p in cited if _quote_in(nq, norm_pass[id(p)])), None)
        if home is None:
            return "quote_not_verbatim"
        if home not in used:
            used.append(home)
    return used


def _figure_problem(ans: str, question: str, quotes: list, used: list) -> str | None:
    qjoined = " ".join(quotes)
    have = _numbers(qjoined) | _spelled(qjoined) | _numbers(question)
    context = set()
    for p in used:
        context |= {_canon(x) for x in re.findall(r"\d+", p["date"] or "")} | _numbers(p["text"]) | _numbers(p["title"])
    for num in _numbers(ans):
        shown = num in context and (_is_year(num) or (num.isdigit() and 0 < int(num) <= MAX_DAY_OF_MONTH))
        if num not in have and not shown:
            return f"number_not_in_quote:{num}"
    for num in _numbers(question):
        if num not in have and not _is_year(num):
            return f"question_number_absent:{num}"
    return None


def _capitalised_runs(sentence: str):
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z0-9&'’-]*", sentence)]
    run: list[str] = []
    prev_end = -9
    for i, (w, st, en) in enumerate(words):
        base = re.sub(r"['’]s$", "", w)
        low = base.lower()
        cap = (
            w[0].isupper()
            and not (w.isupper() and len(w) <= 6)
            and low not in _SOFT
            and low not in _ACRO
            and low not in _COMMON_CAP
        )
        if cap and st - prev_end <= 1 and run:
            run.append(base)
        else:
            if run:
                yield run
            starts_name = i > 0 or (i + 1 < len(words) and words[i + 1][0][0].isupper())
            run = [base] if cap and starts_name else []
        prev_end = en
    if run:
        yield run


def _pieces_known(run: list, pall: str, asked: str, pwords: set) -> bool:
    pieces, cur = [], []
    for w in (x.lower() for x in run):
        if cur and " ".join(cur + [w]) not in pall and " ".join(cur + [w]) not in asked:
            pieces.append(cur)
            cur = []
        cur.append(w)
    pieces.append(cur)
    return all(" ".join(c) in pall or " ".join(c) in asked or (len(c) == 1 and c[0] in pwords) for c in pieces)


def _name_is_known(run: list, pall: str, asked: str, pwords: set, qnorm: str) -> bool:
    phrase = " ".join(w.lower() for w in run)
    if phrase in pall or phrase in asked:
        return True
    if len(run) == 1:
        w = run[0].lower()
        return w in pwords or w + "s" in pwords or (w.endswith("s") and w[:-1] in pwords)
    return (all(w.lower() in pwords for w in run) and phrase in qnorm) or _pieces_known(run, pall, asked, pwords)


def _name_problem(ans: str, question: str, passages: list, quotes: list) -> str | None:
    qnorm = norm_text(" ".join(quotes))
    asked = norm_text(question)
    pall = norm_text(" ".join(p["text"] + " " + p["title"] for p in passages))
    pwords = set(re.findall(r"[a-z0-9]+", pall)) | set(re.findall(r"[a-z0-9]+", asked))
    for sent in SENT_BREAK.split(ans):
        for run in _capitalised_runs(sent):
            if not _name_is_known(run, pall, asked, pwords, qnorm):
                return f"name_not_in_quote:{run[0]}"
    return None


def _question_name_problem(question: str, passages: list) -> str | None:
    ptxt = norm_text(" ".join(p["text"] + " " + p["title"] for p in passages))
    qwords = [
        x for w in re.findall(r"[A-Za-z][A-Za-z'’-]+", question) for x in (re.split(r"-", w) if "-" in w else [w])
    ]
    for i, w in enumerate(qwords):
        low = w.lower()
        if (
            i == 0
            or not w[0].isupper()
            or re.match(r"I['’]", w)
            or low in _SOFT
            or low in _DAYS_MONTHS
            or low in _ACRO
            or low in _COMMON_CAP
            or w.isupper()
        ):
            continue
        if low.replace("’s", "").replace("'s", "") not in ptxt:
            return f"question_name_absent:{w}"
    return None


def _stale_problem(ans: str, used: list, passages: list) -> str | None:
    if not used or not all(p["stale"] for p in used):
        return None
    kinds = ("%" if "%" in ans else "") + ("$" if "$" in ans else "")
    for p in (p for p in passages if not p["stale"]):
        if (
            kinds
            and any(re.search(r"\d\s?%" if k == "%" else r"\$\s?\d", p["text"]) for k in kinds)
            and not (_numbers(p["text"]) & _numbers(ans))
        ):
            return "stale_vs_official"
    return None


def _sources(ans: str, question: str, quotes: list, used: list, passages: list, norm_pass: dict) -> list:
    srcs, seen = [], set()
    for p in used:
        if p["url"] not in seen:
            seen.add(p["url"])
            srcs.append(
                SourceRef(
                    p["url"],
                    p["snapshot"],
                    next((qt for qt in quotes if norm_text(qt) in norm_pass[id(p)]), "")[:SOURCE_QUOTE_CHARS],
                )
            )
    figures = _numbers(ans) - _numbers(question) - {n for n in _numbers(ans) if _is_year(n)}
    for p in passages:
        if figures and len(srcs) < MAX_SOURCES and p["url"] not in seen and figures <= _numbers(p["text"]):
            seen.add(p["url"])
            srcs.append(SourceRef(p["url"], p["snapshot"], ""))
    return srcs


def ground(out: dict | None, passages: list[dict], question: str) -> tuple[Result | None, str]:
    """(Result, 'ok') for a reply that passes every check, else (None, reason)."""
    fields = _reply_fields(out)
    if isinstance(fields, str):
        return None, fields
    ans, quotes, urls = fields
    wanted = {u.rstrip("/") for u in urls}
    cited = [p for p in passages if p["url"].rstrip("/") in wanted]  # several passages can come from one page
    if not cited:
        return None, "no_cited_passage"
    norm_pass = {id(p): norm_text(p["text"]) for p in passages}
    used = _quote_homes(quotes, cited, norm_pass)
    if isinstance(used, str):
        return None, used
    why = (
        _figure_problem(ans, question, quotes, used)
        or _name_problem(ans, question, passages, quotes)
        or _question_name_problem(question, passages)
        or _stale_problem(ans, used, passages)
    )
    if why:
        return None, why
    return Result(ans, _sources(ans, question, quotes, used, passages, norm_pass), True, None, quotes=quotes), "ok"


# ---------------------------------------------------------------------------------------------------- comparing with
# the extractive candidate
def same_evidence(candidate: str, res: Result, cand_sources=()) -> str | None:
    """Why a reader answer adds nothing over the extractive candidate ('same_evidence' | 'subset')."""
    cand_norm = norm_text(candidate)
    ctoks = set(tokenize(candidate))
    for src in cand_sources:
        if getattr(src, "quote", None):
            cand_norm += " " + norm_text(src.quote)
            ctoks |= set(tokenize(src.quote))
    rn, cn = _numbers(res.text) - _years(res.text), (_numbers(cand_norm) | _spelled(cand_norm)) - _years(cand_norm)
    if rn - cn and cn - rn - _spelled(res.text):
        return None
    for qt in getattr(res, "quotes", None) or []:
        nq = norm_text(qt)
        if len(nq) >= 12 and (nq in cand_norm or nq[:60] in norm_text(candidate)):
            return "same_evidence"
        qtoks = set(tokenize(qt))
        if qtoks and len(qtoks & ctoks) / len(qtoks) >= 0.5:
            return "same_evidence"
    cap = lambda t: {w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", t)} - _COMMON_CAP - _SOFT
    if _numbers(res.text) <= _numbers(candidate) and cap(res.text) <= (
        cap(candidate) | {w.lower() for w in re.findall(r"[A-Za-z]+", candidate)}
    ):
        return "subset"
    return None
