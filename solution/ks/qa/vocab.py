"""Word lists and rewrite tables of the text path (retrieval, scoring, refusal checks)."""
from __future__ import annotations

import re

from ..core.tokens import tokenize as _base_tokenize
from ..site_profile import INSTITUTION

# interchangeable words, so 'who qualifies' matches 'you're eligible' and 'shut' matches 'close'
CANON = {
    "qualify": "eligible",
    "qualifying": "eligible",
    "eligibility": "eligible",
    "price": "cost",
    "costs": "cost",
    "shut": "close",
    "closing": "close",
    "closure": "close",
    "closed": "close",
    "closes": "close",
    "mailing": "mail",
    "membership": "member",
    "recognized": "recognize",
    "allowed": "eligible",
    "permitted": "eligible",
    "recognizes": "recognize",
    "recognizing": "recognize",
    "recognition": "recognize",
    "borrow": "loan",
    "borrowing": "loan",
    "borrowed": "loan",
    "borrower": "loan",
    "honored": "honor",
    "pooled": "pool",
    "pooling": "pool",
    "pools": "pool",
    "contributed": "pool",
    "contributes": "pool",
    "contribute": "pool",
    "contributing": "pool",
    "honors": "honor",
    "named": "name",
    "ranked": "rank",
    "ranks": "rank",
    "received": "award",
    "receive": "award",
    "receives": "award",
    "got": "award",
    "granted": "award",
    "awarded": "award",
    "awards": "award",
    "awarding": "award",
}


def tokenize(text: str) -> list[str]:
    """Search tokens with the interchangeable words above mapped to one form."""
    return [CANON.get(t, t) for t in _base_tokenize(text)]


# generic question wording -> extra search terms
QUERY_EXPANSIONS = [
    (re.compile(r"\b(still open|shut|closing|closure|closed|close)\b", re.I), "close closing closed"),
    (re.compile(r"\b(hours|open)\b", re.I), "hours lobby drive-up"),
    (
        re.compile(
            r"\b(join|become a member|membership|eligib\w+|qualif\w+|who (?:else )?can|allowed to|permitted to|able "
            r"to (?:join|become))\b",
            re.I,
        ),
        "join member eligible",
    ),
    (
        re.compile(r"\b(mailing address|po box|p\.o\. box|send (?:a )?(?:mail|check|payment))\b", re.I),
        "address P.O. Box contact",
    ),
    (re.compile(r"\b(lost|stolen)\b", re.I), "lost stolen report"),
    (re.compile(r"\b(college|tuition|university|school|education)\b", re.I), "student"),
]
# question forms whose synonyms are added to the question's terms when scoring sentences
SCORE_SYNONYMS = [
    (re.compile(r"\b(?:put|chip(?:ped)?|threw|kick(?:ed)?) in\b|\bcontribut\w+|\bpooled?\b", re.I), "pool"),
    (
        re.compile(
            r"\b(?:original|founding|first|initial)\s+(?:\w+\s+)?(?:members?|employees|people|staff|investors"
            r"|depositors)\b",
            re.I,
        ),
        "employees founding",
    ),
    (
        re.compile(r"\b(?:start|begin|launch|establish|found|create)\w*\b.*\b(?:credit union|cooperative)\b", re.I),
        "form",
    ),
    (re.compile(r"\b(?:become|join|be) (?:a )?(?:member|part)\b|\bmembership\b", re.I), "eligible"),
    (re.compile(r"\bwho (?:else )?can (?:join|become)\b|\bwho (?:is|are) (?:eligible|allowed)\b", re.I), "eligible"),
]
# groups of interchangeable words used when matching link anchors / URLs to a question
LINK_SYNONYMS = [
    {"join", "member", "membership", "become", "enroll", "signup", "sign"},
    {"rate", "rates", "apy", "apr", "dividend"},
    {"locator", "locate", "location", "branch", "atm", "find", "contact"},
    {"apply", "application", "open"},
    {"login", "log", "sign", "online", "shine"},
]
# words that make a question word 'unknown' to the site without meaning anything about the topic
OK_UNKNOWN = frozenset(
    """wondering anyone anything everything something someone please thanks thank hello would could should might maybe
actually really basically definitely probably currently recently quickly""".split()
)
# question words that carry no topic
STOP_QUESTION_WORDS = (
    frozenset(
        """what whats is are the a an of to for do does can i my me you your we our how where when who which tell about
please link url website web page site find see look view get give share there any in on at it be would could
should im ive going option options offer offers offered""".split()) | {INSTITUTION.lower()}
)
# headings of a product hub page that are navigation or marketing, not products
HEADING_SKIP = re.compile(
    r"\b(?:why|faq|questions?|ready|how|apply|rates?|about|started|love|next|help|learn|calculator|resources?|tools?"
    r"|benefits?|compare|more|"
    r"contact|disclosure|insurance|protection|other|also|wait|there|your|you|we|our|powered|partnership|it works)\b",
    re.I,
)
PRODUCT_WORD = re.compile(
    r"\b(?:loans?|accounts?|cards?|checking|savings|certificates?|mortgages?|credit|visa|ira|lines?|money market"
    r"|share|club|deposit|equipment|insurance)\b",
    re.I,
)
# words of an answer-type question ('what are the hours') that the answer satisfies by its type, not by repeating them
ANSWER_TYPE_WORDS = frozenset(
    {
        "hour",
        "time",
        "date",
        "phone",
        "number",
        "address",
        "cost",
        "fee",
        "limit",
        "amount",
        "many",
        "much",
        "day",
        "price",
        "general",
        "typical",
        "usual",
        "normal",
        "standard",
        "overall",
        "each",
        "zip",
        "postal",
        "code",
        "street",
        "st",
        "avenue",
        "max",
        "maximum",
        "most",
        "least",
        "min",
        "minimum",
    }
)
# words of a site-level statistic question that name no further topic
STAT_WORDS = (
    frozenset(
        """how many much number total count size big large currently now does do has have operate operates serve serves
serving hold holds manage manages credit union cu dollars worth overall altogether all combined people person
bank banking can i is are there at in of a an the full service branch branches location locations office
offices place places member members membership employee employees staff team work works working belong belongs
account holders customers asset assets county counties field present located across total year when what which
was did get got started founded established began begin chartered formed created incorporated cover covers
covered""".split()
    )
    | {INSTITUTION.lower()}
)
# nouns that name one group of things on the site ('branches' = locations = offices)
NOUN_GROUPS = (
    {"branch", "branche", "location", "office", "locations"},
    {"employee", "staff", "colleague"},
    {"member", "customer"},
    {"atm"},
)


def question_terms(raw: str) -> list[str]:
    """The question's search terms: canonical tokens without question words, each once, in order."""
    toks = [t for t in tokenize(re.sub(r"[?!.,;:]", " ", raw)) if t not in STOP_QUESTION_WORDS]
    return list(dict.fromkeys(toks))


def search_text(raw: str) -> str:
    """The question plus the extra terms its wording implies."""
    extra = " ".join(add for rx, add in QUERY_EXPANSIONS if rx.search(raw))
    return f"{raw} {extra}".strip()
