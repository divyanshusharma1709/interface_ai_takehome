"""Small text primitives shared by ingest, answering and conflict mining: plural stemming, Markdown stripping,
sentence breaks, number words."""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------------------------------- plural stemming
def strip_plural(w: str) -> str:
    """'rates' -> 'rate'; leaves short words and 'ss' / 'us' / 'is' endings alone ('business', 'bonus'."""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")) else w


def stem(w: str) -> str:
    """`strip_plural` that also maps '-ies' to '-y' ('policies' -> 'policy'): the search tokenizer's stemmer."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    return strip_plural(w)


def drop_final_s(w: str) -> str:
    """The crudest singular: any trailing 's' goes ('business' -> 'busines')."""
    return w[:-1] if w.endswith("s") else w


# ---------------------------------------------------------------------------------------------------- Markdown
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](src)
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](target): group 1 is the text
MD_LINK_URL = re.compile(r"\[([^\]]*)\]\(([^)\s]*)\)")


def strip_md_links(text: str) -> str:
    """Markdown links reduced to their text."""
    return MD_LINK.sub(r"\1", text)


# ---------------------------------------------------------------------------------------------------- sentence breaks
# Each variant is a regex to `.split()` on; they differ in what may follow the end mark. Use the plainest one that is
# correct for the caller.
SENT_BREAK = re.compile(r"(?<=[.!?])\s+")
SENT_BREAK_OR_LINE = re.compile(r"(?<=[.!?])\s+|\n+")
SENT_BREAK_CAPITAL = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])")
SENT_BREAK_CAPITAL_DIGIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$(])")  # ... or a digit / dollar sign
SENT_BREAK_ABBREV = re.compile(
    r"(?<=[.!?])(?<!\bP\.O\.)(?<!\bSt\.)(?<!\bMt\.)(?<!\bU\.S\.)(?<!\bInc\.)(?<!\bNo\.)(?<!\bext\.)(?<!\bExt\.)"
    r"(?<!\bDr\.)(?<!\bJan\.)(?<!\bFeb\.)(?<!\bAug\.)(?<!\bSept\.)(?<!\bOct\.)(?<!\bNov\.)(?<!\bDec\.)"
    r"\s+(?!(?:[CEMP][SD]?T)\b)(?=[A-Z0-9“\"'(])"
)

# ---------------------------------------------------------------------------------------------------- number words
NUMBER_UNITS = {
    w: i + 1
    for i, w in enumerate(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
        "eighteen nineteen".split()
    )
}
NUMBER_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
# 'forty-two', 'seventeen', 'hundred' as one regex fragment
NUMBER_WORD_PATTERN = (
    "(?:"
    + "|".join([*NUMBER_UNITS, *NUMBER_TENS, "hundred"])
    + ")(?:[- ](?:"
    + "|".join(list(NUMBER_UNITS)[:9])
    + "))?"
)


# ---------------------------------------------------------------------------------------------------- phone numbers
PHONE_ANY_SEP = re.compile(r"\b\d{3}[-.)\s]+\d{3}[-.\s]\d{4}\b")  # 800-373-7600, (800) 373 7600, 800.373.7600
PHONE_DASHED = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
