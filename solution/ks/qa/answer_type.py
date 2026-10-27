"""What kind of answer a question asks for (hours, an address, a count, a date, a phone line, a rate ...)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.dates import MONTH_WORD
from ..core.textutil import NUMBER_WORD_PATTERN, PHONE_ANY_SEP
from ..site_profile import INSTITUTION

NUMBER_WORDS = NUMBER_WORD_PATTERN

# ---------------------------------------------------------------------------------------------------- shapes of an
# answer sentence
DATE_RX = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.I,
)
STREET_RX = re.compile(
    r"\b\d{2,6}\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*)*\s+(?:St|Ave|Rd|Dr|Ln|Blvd|Way|Ct|Pkwy|Hwy|Street|Avenue|Road|Drive"
    r"|Lane)\b|\b(?i:P\.?\s?O\.?\s+Box\s+\d+)"
)
HOURS_RX = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b|\bnoon\b", re.I)
NUMBER_RX = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
CLOSING_RX = re.compile(r"\b(clos\w+|shut\w*|open\w*|ending|end(?:s|ed)?|consolidat\w+)\b", re.I)
_COUNT_RX = re.compile(r"\d|\b(?:" + NUMBER_WORDS + r"|dozen|thousand|million)\b", re.I)
_DATEISH_RX = re.compile(
    rf"\b{MONTH_WORD}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s*(?:19|20)\d\d)?|\b(?:19|20)\d\d\b|\b\d{{1,2}}/\d{{1,2}}(?:"
    rf"/\d{{2,4}})?\b",
    re.I,
)
_MONEYISH_RX = re.compile(
    r"\$\s?\d[\d,.]*(?:\s?(?:million|thousand|billion|[KkMm])\b)?|\b\d[\d,.]*\s?(?:%|percent)", re.I
)
_ORDINAL_RX = re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|once|twice)\b", re.I)
_AMOUNT_RX = re.compile(
    r"\$\s?\d|\d[\d,.]*\s?(?:%|percent|dollars)|\bno\b[^.]{0,25}\b(?:charges?|fees?|costs?)\b|\bfree\b|\bwaived\b", re.I
)
_FIGURE_RX = re.compile(
    r"\$\s?\d|\d\s?%|\d\s?percent|\bno\b[^.]{0,30}\b(?:fees?|charge|cost|minimum)\b|\bfree\b|\bwaived?\b", re.I
)
_URL_RX = re.compile(r"https?://\S+")


class _CountDetail:
    """A count answer type: a number or number word that is not just a date ('Jan."""

    @staticmethod
    def search(s: str):
        return _COUNT_RX.search(_MONEYISH_RX.sub(" ", _DATEISH_RX.sub(" ", s)))


def expected_detail(q):
    raw = q.raw.lower()
    if re.search(r"\bhours?\b|\bwhat time\b|\bopen (?:at|until|from)\b", raw):
        return re.compile(HOURS_RX.pattern + r"|\bclosed\b", re.I)
    # a time of day or a date
    if re.search(r"\bwhen (?:does|do|is|are|will)\b.*\b(?:open|close)\b", raw):
        return re.compile(HOURS_RX.pattern + "|" + DATE_RX.pattern + r"|\bclosed\b", re.I)
    if re.search(
        r"\b(?:born|birth ?(?:day|date|year|place)|died|net worth|salary)\b|\b(?:when|what year|which year|date) "
        r"(?:\w+ ){0,3}(?:death|birth)\b",
        raw,
    ):
        return re.compile(
            r"\b(?:born|birth\w*)\b[^.]{0,60}\b\d{4}\b|\b\d{4}\b[^.]{0,60}\b(?:born|birth)|\b(?:died"
            r"|death)\b[^.]{0,60}\b\d{4}\b"
            r"|\b(?:salary|net worth|compensation)\b[^.]{0,40}\$\s?\d|\$\s?\d[^.]{0,40}\b(?:salary|net worth)\b",
            re.I,
        )
    # a physical size needs a unit
    if re.search(r"\bhow (?:tall|heavy|wide|deep)\b", raw):
        return re.compile(
            r"\b\d[\d,.]*\s?(?:feet|foot|ft|stor(?:y|ies)|floors?|meters?|inches|inch|pounds|lbs|square|sq)\b", re.I
        )
    # a ZIP is part of a street address
    if re.search(r"\bzip(?: ?code)?\b|\bpostal\b|\bstreet address\b", raw):
        return STREET_RX
    if re.search(
        r"\b(?:charter|licen[sc]e|nmls|certificate|policy|registration|identification|id|account)\s+(?:number|no\.?|#)",
        raw,
    ):
        # an identifier is digits: 'call us' is not the number
        return re.compile(r"(?<![\d-])\d{4,}(?![\d-])")
    if re.search(r"\bphone number\b|\bnumber to call\b|\bwhat number\b|\bhow (?:do|can) i (?:call|phone)\b", raw):
        return PHONE_ANY_SEP
    if re.search(r"\bhow many times\b|\bhow often\b", raw):
        # an ordinal answer type ('for the third time')
        return re.compile(_COUNT_RX.pattern + "|" + _ORDINAL_RX.pattern, re.I)
    if re.search(r"\bhow many\b|\bnumber of\b", raw):
        return _CountDetail()
    if re.search(r"\blimits?\b|\bmaxi?mum\b|\bmax\b|\bcap\b|\bhow high\b|\bat most\b", raw):
        return re.compile(
            r"(?=.*\b(?:limits?|maxi?mum|max|cap|up to|at most|no more than|not exceed|exceed\w*)\b)"
            r"(?=.*(?:\$\s?\d|\d[\d,]*\s?(?:times|transactions|per (?:day|month)|dollars|%|percent)))",
            re.I | re.S,
        )
    if re.search(r"\bhow much\b|\b(?:cost|price|fee|charge)\b", raw):
        return _AMOUNT_RX
    if re.search(r"\bwhat (?:date|day)\b|\bon what date\b", raw):
        return DATE_RX
    return None


def has_expected_detail(q, text: str) -> bool:
    """Does `text` (links ignored) hold the kind of detail the question asks for?"""
    need = expected_detail(q)
    return need is None or bool(need.search(_URL_RX.sub("", text)))


def shows_expected_detail(q, text: str) -> bool:
    """Does `text` (links ignored) hold a kind of detail the question asks for?"""
    need = expected_detail(q)
    return need is not None and bool(need.search(_URL_RX.sub("", text)))


def amount_predicate(q):
    raw = q.raw.lower()
    if not re.search(r"\bhow much\b|\bminimum\b|\bcost|\bfee\b|\bprice\b|\bcharge", raw):
        return None
    groups = []
    if re.search(
        r"\bopen\w*\b|\bjoin\w*\b|\bminimum\b|\bdeposit\w*\b|\bstart\w*\b|\bfund\w*\b|\binitial\b|\bbecome\b", raw
    ):
        groups.append(r"open\w*|join\w*|deposit\w*|minimum|balance|at least|or more|initial|fund\w*|start\w*|share")
    if re.search(r"\bcost|\bfee\b|\bfees\b|\bprice\b|\bcharge|\bpay\b", raw):
        groups.append(
            r"fee|fees|cost\w*|charge\w*|price\w*|per (?:item|month|year|check|day|transaction)|monthly|annual|waive\w*"
        )
    if re.search(r"\bborrow|\bloan amount|\bfinance|\bmaximum|\bmax\b|\bat most|\bup to\b", raw):
        groups.append(r"up to|maximum|max|borrow\w*|finance\w*|loan amount")
    return re.compile(r"\b(?:" + "|".join(groups) + r")\b", re.I) if groups else None


def has_figure(q, sent: str) -> bool:
    """Does `sent` carry the figure a rate / fee question asks for?"""
    if not _FIGURE_RX.search(sent):
        return False
    if q.kind == "fee" and not re.search(r"\bfees?\b|\bcharges?\b|\bcosts?\b|\bfree\b", sent, re.I):
        return False
    if q.kind == "rate" and not re.search(r"\b(?:rate|apr|apy|dividend|interest|yield)\b", sent, re.I):
        return False
    if q.kind == "rate" and not re.search(r"\d\s?%|\d\s?percent", sent):
        return False
    return True


# ---------------------------------------------------------------------------------------------------- sentence score
# bonus
# Each rule: (does the question ask for this?, does the sentence have it?, bonus). Rules add up in this order.
_ANNUAL_FEE_Q = re.compile(r"\bannual\b.*\bfees?\b|\bfees?\b.*\bannual\b")
_WHEN_Q = re.compile(r"\bwhen\b|\bwhat (?:day|date)\b|\bdeadline\b|\bby what\b")
_TIME_Q = re.compile(r"\bwhat time\b|\bdate and time\b|\btime and (?:date|day)\b|\bday and time\b|\bwhen\b.*\bstart")
_COUNT_Q = re.compile(r"\bhow (?:many|much)\b|\bnumber of\b|\btotal\b")
_HOW_MANY_NOUN = re.compile(r"\bhow many (\w{3,})")
_HOW_MUCH_Q = re.compile(r"\bhow much\b")
_ADDRESS_Q = re.compile(r"\baddress\b|\bzip\b|\bpostal\b|\bwhere is\b|\blocated\b|\bwhere\b.*\bbranch\b")
_HOURS_Q = re.compile(r"\bhours?\b|\bopen\b|\bclose\b|\blobby\b|\bdrive[- ]?up\b|\bwhat time\b")
_CLOSING_Q = re.compile(r"\b(still open|closing|closure|closed|close|shut)\b")
_PHONE_Q = re.compile(r"\bphone\b|\bcall\b|\bnumber\b")
_HOWTO_Q = re.compile(r"\bhow (?:do|can|to)\b|\bwhat do i need\b|\bwhat should i do\b")
_ACTION_WORD = re.compile(r"\b(call|visit|log in|sign in|open|go to|click|select|contact|submit)\b", re.I)
_APR_WORDS = re.compile(r"annual percentage", re.I)


class TypeBoost:

    def __init__(self, q):
        raw = q.raw.lower()
        self.annual_fee = bool(_ANNUAL_FEE_Q.search(raw))
        self.when = bool(_WHEN_Q.search(raw))
        self.time = bool(_TIME_Q.search(raw))
        self.count = bool(_COUNT_Q.search(raw))
        noun = _HOW_MANY_NOUN.search(raw)
        self.counted = (
            re.compile(rf"\b(?:\d[\d,]*|{NUMBER_WORDS})\s+(?:[\w-]+\s+){{0,2}}{re.escape(noun.group(1)[:5])}", re.I)
            if noun
            else None
        )
        self.names_institution = INSTITUTION.lower() in raw
        self.how_much = bool(_HOW_MUCH_Q.search(raw))
        self.address = bool(_ADDRESS_Q.search(raw))
        self.hours = bool(_HOURS_Q.search(raw))
        self.closing = bool(_CLOSING_Q.search(raw))
        self.phone = bool(_PHONE_Q.search(raw))
        self.howto = bool(_HOWTO_Q.search(raw))

    def score(self, s: str) -> float:
        b = 0.0
        if self.annual_fee and _APR_WORDS.search(s):
            # 'annual fee' is not 'annual percentage rate'
            b -= 0.6
        if self.when and DATE_RX.search(s):
            b += 0.35
        if self.time and HOURS_RX.search(s):
            # 'date and time': the sentence that gives both
            b += 0.45 + (0.3 if DATE_RX.search(s) else 0.0)
        if self.count and NUMBER_RX.search(s):
            b += 0.25
        if self.counted is not None and self.counted.search(s):
            # '22 branches' answers 'how many branches'
            b += 0.4
            if self.names_institution and INSTITUTION.lower() in s.lower():
                b += 0.25
        if self.how_much and re.search(r"\$\s?\d", s):
            b += 0.45
        if self.address and STREET_RX.search(s):
            b += 0.6
        if self.hours and HOURS_RX.search(s):
            b += 0.35
        if self.closing and CLOSING_RX.search(s):
            b += 0.35 + (0.25 if DATE_RX.search(s) else 0)
        if self.phone and PHONE_ANY_SEP.search(s):
            b += 0.3
        if self.howto and _ACTION_WORD.search(s):
            b += 0.15
        return b


# ---------------------------------------------------------------------------------------------------- branch table
@dataclass(frozen=True)
class BranchAsk:
    """Which slots of a branch record a question asks for."""
    hours: bool
    address: bool
    phone: bool
    status: bool
    services: bool
    # the question says branch / location / office / lobby / drive-up
    about_branch: bool
    # asks why or what happens: the article answers, not the table
    explanation: bool

    @property
    def any_detail(self) -> bool:
        return self.hours or self.address or self.phone or self.status


def branch_ask(low: str) -> BranchAsk:
    return BranchAsk(
        hours=bool(
            re.search(
                r"\bhours?\b|\bopen\b|\bclose[sd]?\b|\blobby\b|\bdrive[- ]?up\b|\bwhat time\b|\bsaturday\b|\bsunday\b"
                r"|\bweekends?\b",
                low,
            )
        ),
        address=bool(re.search(r"\baddress\b|\bwhere is\b|\blocated\b|\bzip\b|\bpostal\b|\bstreet\b", low)),
        phone=bool(re.search(r"\bphone\b|\bcall\b|\bnumber\b|\bcontact\b|\breach\b", low))
        and not re.search(r"\bhow many\b|\bnumber of (?!the\b)\w+s\b", low),
        status=bool(
            re.search(
                r"\bstill\b|\bclos(?:e|ed|ing|ure)\b|\bshut|\bopen(?:ed)? (?:yet|now|today)\b|\bis .* open\b", low
            )
        ),
        services=bool(
            re.search(
                r"\bservices?\b|\boffer\b|\bamenit|\bhave a\b|\bdrive[- ]?up window\b|\batm\b|\bvideo teller\b", low
            )
        ),
        about_branch=bool(re.search(r"\bbranch(?:es)?\b|\blocations?\b|\boffices?\b|\blobby\b|\bdrive[- ]?up\b", low)),
        explanation=bool(
            re.search(
                r"\bwhy\b|\bhow come\b|\breasons?\b|\bwhat (?:will )?happens?\b|\bwhat about\b|\bwill my\b|\bwhere "
                r"(?:should|do|can) i go\b|\binstead\b"
                r"|\bwhat should i do\b|\bhow (?:do|will|can|would) (?:i|my)\b|\bexplain\b",
                low,
            )
        ),
    )


# ---------------------------------------------------------------------------------------------------- phone and rate
# asks
_PHONE_ASK = re.compile(r"\bphone\b|\bcall\b|\bnumber\b|\btelephone\b|\bdial\b|\bhotline\b|\bextension\b|\bext\b")
_NOT_A_PHONE_ASK = re.compile(r"\bhow many\b|\bnumber of\b|\brouting\b|\baccount number\b")
RATE_ASK = re.compile(r"\b(?:rates?|apy|apr|dividends?|interest)\b", re.I)
# asks for an amount, not for a rate value
AMOUNT_ASK = re.compile(r"\b(?:borrow|limit|maximum|max|cash back|how much can)\b", re.I)
_GENUINE_RATE_ASK = re.compile(
    r"\b(fees?|charges?|costs?|price|rates?|aprs?|apys?|interest|dividends?|penalt\w+|percent)\b|%", re.I
)
_NOT_A_RATE_ASK = re.compile(r"\b(average|maximum|max|how many|number of|total|how long|when|why|who|limit)\b", re.I)


def wants_phone_line(low: str) -> bool:
    """A phone / extension question as the lost-card line lookup reads it (not a count, a routing or account number)."""
    return bool(_PHONE_ASK.search(low)) and not _NOT_A_PHONE_ASK.search(low)


def genuine_rate_ask(q) -> bool:
    """A real rate / fee question: the answer sentence must carry the figure."""
    return q.kind in ("rate", "fee", "min_open", "min_earn") and q.intent in ("value", "list", "change") and bool(
        _GENUINE_RATE_ASK.search(q.raw)) and not _NOT_A_RATE_ASK.search(q.raw)
