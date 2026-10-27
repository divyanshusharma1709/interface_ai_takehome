"""The one website the miners are pointed at: product vocabulary, URL layout, page wording."""
from __future__ import annotations

import re

from ..core.dates import MONTH_WORD

# ---- rate conflicts (numeric.py, merge.py, dates.py)
RATE_ATTRS = ("apy", "rate", "apr", "dividend_rate")           # attributes of a rate fact, headline first
FAMILY_SALIENCE = {"mortgage": 1.0, "auto_loan": 1.0, "checking": 0.95, "certificate": 0.95, "savings": 0.9,
                   "money_market": 0.9, "credit_card": 0.85, "personal_loan": 0.85, "heloc": 0.85,
                   "home_equity_loan": 0.85, "ira": 0.75}      # how much a member cares about a product family's rates
# appended to a product name that lacks it
FAMILY_WORD = {"auto_loan": "auto loan", "certificate": "certificate (CD)", "mortgage": "mortgage"}
# (headline, twin, topic tag, label)
RATE_PAIRS = (("rate", "apr", "rate/APR", "APR"), ("apy", "dividend_rate", "APY/dividend rate", "dividend"))
JUMBO_WORD = "jumbo"                                           # jumbo mortgages are listed after the standard ones
IRA_ROW_FILLER = frozenset({"ira", "certificate", "cd"})       # words that do not tell one special from another
IRA_TYPE_PAIR_LABEL = "Traditional vs Roth"

# ---- program periods and deadlines (periods.py)
PERIOD_DATE = re.compile(r"(" + MONTH_WORD + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d\d))?")
PERIOD_CUE = re.compile(r"application period|application deadline|deadline|apply by|applications? (?:are |is "
                        r")?due|period begins|opens|closes", re.I)
PERIOD_START_WORDS = r"begins|opens|starting|open "
# words that do not name a program
PERIOD_GENERIC_WORDS = frozenset("""the a an of to for and in on application applications period deadline program grant
dupaco community begins open opens closed apply now scroll review guidelines webinar presentation pdf
powered by""".split())


# a worked payment example ("If you borrow $X at Y% ... payment obligation would be $Z"): illustrative, not a statistic
PAYMENT_EXAMPLE = re.compile(
    r"(?:(?P<what>[A-Z][A-Za-z&/ ]{2,30}?)\s+(?:loan\s+)?payment example[^:]*:\s*)?If you "
    r"borrow(?:ed)?\s+\$(?P<amt>[\d,]+)\s+at\s+an?\s+(?P<apr>[\d.]+)%"
    r".{0,60}?with an?\s+(?P<term>\d+)[- ]month term.{0,60}?payment obligation would be \$(?P<pay>[\d,]+\.\d\d)", re.S)

# ---- repeated statistics (slots.py, stats.py): the boilerplate statistics of the institution's own pages
STATIVE_PAST_WORDS = frozenset({"founded", "chartered", "established", "limited", "insured", "based", "united",
    "required", "licensed"})
# ^ past-tense words that still state a standing fact; any other '-ed' word marks an event (differs by year)
# identifiers, not figures
NUMBER_SKIP_BEFORE = frozenset({"ext", "extension", "nmls", "no", "x", "page", "step", "version", "ste", "suite"})
# street numbers
STREET_WORD = re.compile(r"^(?:st|street|ave|avenue|rd|road|blvd|dr|drive|ln|lane|way|ct|hwy|pkwy|sw|se|"
                         r"ne|nw|n|s|e|w)\b",
                         re.I)
# 'for the fifth consecutive year'
ORDINAL_CUES = frozenset({"time", "times", "year", "years", "consecutive", "straight", "row"})
# an ordinal counts only as a ranking streak ('the fourth time')
ORDINAL_CLAIM_WORDS = ("time", "times", "consecutive", "straight")
STATE_NOUNS = frozenset({"branch", "branches", "employees", "members", "membership", "memberowners", "counties",
    "county", "assets",
                         # nouns of the institution's own statistics
                         "atms", "locations", "offices", "states", "years", "customers", "staff"})
# one claim however the sentence around it is worded
NOUN_GROUP = frozenset({"employees", "members", "counties"})
BRANCH_WORDS = ("branch", "branches")
STAT_ALIASES = [(re.compile(r"\bbranch office locations?\b"), "branches"), (re.compile(r"\bmember-?owners?\b"),
    "members")]   # (wording, plain word)
MEMBERS_CUES = frozenset({"membership", "members", "grown"})
STAT_TOPICS = {"members": "Number of members", "assets": "Total assets", "employees": "Number of employees",
               # the statistics the official statistics box also states
               "branches": "Number of branch locations"}
COUNTIES_TOPIC = "Number of counties served"
ASSETS_WORD = "assets"                                         # a dollar figure around this word is the total assets
COUNTED_NOUNS = ("employees", "members")                       # nouns whose count is a statistic on its own
