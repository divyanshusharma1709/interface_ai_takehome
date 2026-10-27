"""The search tokenizer, shared by indexing and querying."""

from __future__ import annotations

import re
import unicodedata

from .dates import dates_mentioned
from .textutil import MD_LINK_URL, stem

_TOKEN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?|[^\W\d_]+(?:['’][^\W\d_]+)?")
_SLASH = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])")
_STOP = frozenset("""a an and are as at be by can do does for from has have how i if in is it its me my of on or our
so than that the their them then there these they this to us was we were what when where which who why will with
would you your please tell about get give provide know like want need much many any""".split())
_SYNONYMS = {"cd": "certificate", "cds": "certificate", "certificates": "certificate"}
_URL = re.compile(r"https?://(?:www\.)?([^\s)\]]+)")


def _number(tok: str) -> list[str]:
    n = tok.lstrip("$").replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    out = [n]
    if n.endswith(".99") and "." in n:
        out.append(n[:-3])
    return out


def tokenize(text: str) -> list[str]:
    """Lowercased, accent-folded word/number tokens (+ 'n/n' and ISO-date tokens): numbers canonical."""
    out: list[str] = []
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    out.extend(f"{a}/{b}" for a, b in _SLASH.findall(text))  # joint token; digits stay separate too
    out.extend("d" + d for d in dates_mentioned(text, cap=1000))  # 09/01/2026 == September 1, 2026
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        if tok[0].isdigit() or tok[0] == "$":
            out.extend(_number(tok))
            continue
        tok = re.sub(r"['’]s$", "", tok).replace("'", "").replace("’", "")
        if tok in _STOP or len(tok) < 2 and not tok.isdigit():
            continue
        tok = _SYNONYMS.get(tok, tok)
        out.append(stem(tok))
    return out


def index_view(text: str) -> str:
    """Markdown links become 'text url-words' (scheme/www/query dropped) so URL questions can match."""

    def link(m: re.Match) -> str:
        u = _URL.sub(r"\1", m.group(2)).split("?")[0]
        return f"{m.group(1)} {u}" if not u.startswith(("tel:", "mailto:", "#")) else m.group(1)

    return MD_LINK_URL.sub(link, text)
