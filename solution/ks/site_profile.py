"""Everything specific to the one website this system was built for (dupaco.com): names, URL layout, boilerplate and vocabulary.

The mechanisms elsewhere in `ks` are generic; this module is the single place that says which institution they are pointed at.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse
from dataclasses import dataclass

from .core.types import Voice

HOST_SUFFIX = "dupaco.com"                               # pages on this host make the answers speak of the institution by name
INSTITUTION = "Dupaco"
CONTACT_CHANNELS = "by phone, in Shine Online Banking or at a branch"

DEFAULT_VOICE = Voice(INSTITUTION, CONTACT_CHANNELS)

# ---- URL layout
ORIGIN = "https://dupaco.com"
_HOME_URL = re.compile(r"https?://(?:www\.)?dupaco\.com/?")
_OTHER_ORIGIN = re.compile(r"https?://(?!(?:www\.)?(?:\w+\.)?dupaco\.com)")
_ORIGIN_PREFIX = re.compile(r"https?://(?:www\.)?dupaco\.com")
_TITLE_SUFFIX = re.compile(r"\s*\|\s*Dupaco\s*$")

# Page-type rules of `ingest.pages.classify`: which URL paths hold which kind of page.
LEGAL_PATHS = ("/privacy-policy", "/about/disclosures", "/digital-privacy-policy", "/disclosures",
               "/photo-release-form", "/subscription-preferences")
NEWSROOM_INDEX_PATHS = ("/newsroom", "/resources/dupaco-news")
NEWSROOM_LISTING_PREFIXES = ("/newsroom/author", "/newsroom/tag")
NEWSROOM_PREFIX = "/newsroom/"
BLOG_INDEX_PATHS = ("/learn/blog", "/business/resources")
BLOG_LISTING_PREFIXES = ("/learn/blog/tag", "/learn/blog/author", "/resources")
BLOG_PREFIXES = ("/learn/blog/", "/business/resources/", "/dupacodaily/", "/protection/", "/covid-19")
RATE_SHEET_PATHS = ("/rates", "/business-rates")
CONTACT_PATHS = ("/contact", "/connect/contact-us")
BUSINESS_PATH_PREFIX = "/business"
PRODUCT_PREFIXES = ("/personal", "/business", "/checking", "/loans", "/personal-loans", "/savings-account",
                    "/earn-more", "/switch-and-earn", "/membership", "/about/join", "/about/member-services",
                    "/first-time-homebuyer-grant", "/hardship-grant", "/financial-services", "/mortgage-refi",
                    "/homebuying-guidance", "/online-and-mobile-banking", "/connect", "/mobile", "/shine",
                    "/brighttrack", "/manage-account-alerts", "/thankuse", "/rates", "/learn/master-your-money",
                    "/money-makeover", "/scholarships", "/non-profit-grant", "/gift-of-innovation-grant")


def is_business_url(url: str) -> bool:
    """Pages of the business section (a /business path, or a business.* host) speak of business accounts."""
    parts = urlparse(url)
    return parts.path.lower().startswith(BUSINESS_PATH_PREFIX) or parts.netloc.lower().startswith("business")


def is_home_url(url: str) -> bool:
    return _HOME_URL.fullmatch(url) is not None


def links_off_site(text: str) -> bool:
    """True when `text` holds a link to a host other than the institution's own."""
    return _OTHER_ORIGIN.search(text) is not None


def is_own_site(url: str) -> bool:
    return HOST_SUFFIX in url


def path_only(url: str) -> str:
    """The URL without its scheme and host (own-site URLs only)."""
    return _ORIGIN_PREFIX.sub("", url)


def strip_title_suffix(title: str) -> str:
    """'Savings | Dupaco' -> 'Savings'."""
    return _TITLE_SUFFIX.sub("", title)


def voice_for(urls) -> Voice:
    """The institution is named only when the data is its own website; other sites get neutral wording."""
    hosts = {u.split("//")[-1].split("/")[0].lower() for u in urls}
    return DEFAULT_VOICE if any(h.endswith(HOST_SUFFIX) for h in hosts) else Voice(None)


# ---- Vocabulary of the LLM layer (`llm.merge`, `llm.grounding`): the institution's product families and its own names
# The noun that names each product family in the reworded question the parser's fields are turned into (what the fact store parses back).
FAMILY_NOUNS = {"certificate": "certificate", "savings": "savings", "checking": "checking", "money_market": "money market", "credit_card": "credit card",
                "mortgage": "mortgage", "auto_loan": "auto loan", "personal_loan": "personal loan", "heloc": "HELOC", "home_equity_loan": "home equity loan",
                "home_improvement_loan": "home improvement loan", "recreational_loan": "recreational loan", "ira": "IRA", "hsa": "health savings"}
# Words in a question that back an LLM-proposed family: a family the question never hints at is dropped.
FAMILY_HINTS = {
    "certificate": r"\bcds?\b|certificate|term deposit|\bterm\b", "checking": r"check|draft|debit|everyday account|\baccount\b", "savings": r"saving|share|club|holiday|christmas|dividend",
    "money_market": r"money market|\bmm\b|market", "credit_card": r"credit|card|visa|rewards", "mortgage": r"mortgage|home|house|refi|purchase|fixed|\barm\b",
    "auto_loan": r"auto|car|vehicle|truck|motorcycle|\brv\b|boat|camper|used|new", "personal_loan": r"personal|loan|borrow", "heloc": r"heloc|equity|line of credit",
    "home_equity_loan": r"equity", "ira": r"\bira\b|retire|roth|rollover", "hsa": r"\bhsa\b|health saving", "student_loan": r"student|college|school|education",
    "business_loan": r"business|loan", "construction": r"construction|build"}
# Words that are the institution itself or its footprint, not an entity worth boosting retrieval for.
ORG_WORDS_SITE = {"dupaco", "iowa", "illinois", "wisconsin"}
# Capitalised words the answer grounding must not mistake for an invented proper name.
SOFT_WORDS_SITE = {"dupaco", "dupaco's"}
ACRONYMS_SITE = {"iowa"}
COMMON_CAPS_SITE = {"shine", "achieve", "foundation"}                      # product-brand and organisation words the pages use freely
PLACE_WORDS_SITE = {"iowa", "illinois", "wisconsin", "dubuque"}            # the footprint and head-office city: never proof of a name


# ---------------------------------------------------------------------------------------------------- question answering (qa/)
# Branches and places
HOME_CITY = "dubuque"                                    # the city with many branches: a bare city name there does not single one out
STATES = {"Iowa": "IA", "Illinois": "IL", "Wisconsin": "WI", "Wis.": "WI", "Ill.": "IL", "Minnesota": "MN", "Missouri": "MO"}
REGION_WORDS = ("iowa", "illinois", "wisconsin", "dubuque")        # place names that are known without being a branch
LEARNING_LAB = "Learning Lab"                            # a branch the announcements know only by this name
INNOVATION_LAB_WORDS = ("learning", "lab", "innovation")
STORE_BRANCH_WORD = "Hy-Vee"                             # branches inside a supermarket carry the store's name


@dataclass(frozen=True)
class PlaceMention:
    """A place that news posts name in prose (it has no entry in the locator list). `street` / `city` are fixed values, else regex groups 1 / 2."""
    pattern: re.Pattern
    label: str
    kind: str
    state: str
    street: str | None = None
    city: str | None = None
    aliases: tuple = ()
    services: tuple = ()
    first_only: bool = False                             # one source per page, however often the page repeats the sentence


PLACE_MENTIONS = (
    PlaceMention(re.compile(r"The Innovation Lab at (\d{1,6} [A-Z][\w.]*(?: [A-Z][\w.]*)?(?: [WENS]\b\.?)?) in ([A-Z][a-z]+)"),
                 "Innovation Lab", "innovation lab", "IA", services=("live video teller", "coworking space")),
    PlaceMention(re.compile(r"Voices? Building,? (?:located )?(?:at )?(1000 Jackson St)"), "Voices Building", "operations center", "IA",
                 street="1000 Jackson St.", city="Dubuque", aliases=("Operations Center", "Millwork"), first_only=True),
)


@dataclass(frozen=True)
class StoreMention:
    """A post saying where a supermarket branch is: its street fills a missing address and `alias` names it."""
    pattern: re.Pattern                                  # group 1 = the street
    street_template: str
    alias: str


STORE_MENTION = StoreMention(re.compile(r"branch inside the (South Locust St\.) Hy-Vee store in (Dubuque)"), "the Hy-Vee store on {}", "Locust")

# Markup of the locator page and of branch announcements
LOCATOR_BRANCH_MARKER = "Dupaco Branch"                  # line that opens one branch entry in the locator list
LOCATOR_ATM_MARKER = "Surcharge-Free ATM"                # line under an ATM's heading in the locator list
LOCATOR_ATM_PREFIX = "Dupaco ATM @ "
RELEASE_FOOTER = "ABOUT DUPACO"                          # the boilerplate paragraph that closes every press release

# Phone numbers
MAIN_PHONES = frozenset({"8003737600", "5635577600"})    # the institution's main lines: the lost-card line is an extension of one of them

# Product vocabulary
# Member wording -> the wording of this institution's rate sheets, applied in this order to every question before it is parsed.
QUESTION_WORDING = [
    (r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:dollars?|bucks|usd)\b", r"$\1"),
    (r"\bcash[- ]?back\b|\brewards? (?:credit )?card\b|\bpoints? card\b|\btravel card\b", "rewards visa card"),
    (r"\blow[- ](?:rate|interest) (?:credit )?card\b", "platinum visa card"),
    (r"\bbalance[- ]transfer (?:credit )?card\b", "platinum visa balance transfer card"),
    (r"\bcashier'?s? checks?\b|\bbank checks?\b", "official check"),
    (r"\bsavings? (?:interest|dividends?)\b", "savings rate"),
    (r"\bmoney[- ]?market\b", "money market"),
    (r"\bcertificates? of deposits?\b", "certificate"),
    (r"\bboat or rv\b", "boat rv"),
    (r"\bmort?gages?\b", "mortgage"),
]
# Rewrites applied to chat questions before they are cleaned (acronyms the member types, fee names the site uses).
ACRONYM_EXPANSIONS = [(r"\bAD ?& ?D\b", "accidental death and dismemberment")]
PRODUCT_ALIASES = {"share savings": r"\b(?:regular|basic|standard|ordinary|plain)\s+(?:share\s+)?savings\b"}   # product -> what members call it

# Site-level figures
# The institution repeats a few figures in the closing "About" paragraph of every release. Each topic: the words that name it (any wording), and
# the sentence that states it. `NEVER_STATED` marks a topic that only the dated official box can answer.
NEVER_STATED = r"(?!x)x"


@dataclass(frozen=True)
class SiteFigure:
    name: str
    asked: str                                           # regex: how a question names the topic
    stated: str                                          # regex: the sentence of the boilerplate that states it


SITE_FIGURES = (
    SiteFigure("members", r"\b(?:members?|membership|account ?holders|customers|belong\w*|bank(?:s|ing)? with)\b", r"membership (?:has grown to|totals) (?:more than |over )?[\d,]+"),
    SiteFigure("assets", r"\bassets?\b", r"with assets exceeding"),
    SiteFigure("employees", r"\b(?:employees?|staff|workforce|headcount|colleagues|team members?|work(?:s|ing)? (?:at|for)|employs?)\b", r"(?:has|employs)\s+(?:over|more than|nearly)\s+[\d,]+\s+employees"),
    SiteFigure("branches", r"\b(?:branch\w*|locations?|offices?|places? (?:to|where|can)\b|in[- ]person|full[- ]service)", r"[\d,]+\s+branch office locations"),
    SiteFigure("counties", r"\bcount(?:y|ies)\b", r"serves residents in [\d,]+ counties"),
    SiteFigure("deposits", r"\b(?:total )?deposits\b", NEVER_STATED),
    SiteFigure("loans", r"\b(?:total loans|total outstanding loans|loans outstanding|outstanding loans|loan portfolio)\b", NEVER_STATED),
    SiteFigure("founded", r"\b(?:founded|started|established|began|begin|chartered|formed|created|incorporated|get started|got started)\b", r"Founded in 19\d\d"),
)
# Sentences that give a nationwide figure rather than the institution's own (site figures / branch counts)
NATIONAL_FIGURE_MARKERS = r"U\.S\.|nationwide|industry"
NATIONAL_COUNT_MARKERS = r"U\.S\.|nationwide|across the (?:country|nation|United)|industry|nearly 100,000"

# Product families
# family -> (noun in 'our <noun> products: ...', noun in 'current <noun> rates are published at ...'); the second is also what a rates page is called
FAMILY_LIST_NOUNS = {
    "savings": ("savings", "savings account"), "checking": ("checking", "checking"), "credit_card": ("credit card", "credit card"),
    "money_market": ("money market", "money market"), "certificate": ("certificate", "certificate"), "ira": ("IRA", "IRA"),
    "mortgage": ("mortgage", "mortgage"), "auto_loan": ("auto loan", "auto loan"), "personal_loan": ("personal loan", "personal loan"),
    "hsa": ("HSA", "HSA"),
}
RATES_PAGE_DEFAULT_NOUN = "deposit and loan"             # rates page of no particular family
# a question about one family lists the products of these families (savings includes money market and certificates)
FAMILY_GROUPS = {"savings": ("savings", "money_market", "certificate"), "checking": ("checking",), "credit_card": ("credit_card",),
                 "money_market": ("money_market",), "certificate": ("certificate",), "ira": ("ira",), "mortgage": ("mortgage",),
                 "auto_loan": ("auto_loan",), "personal_loan": ("personal_loan",)}
# family -> words of the hub page URL that lists its products
FAMILY_HUB_TOPICS = {"savings": {"savings"}, "checking": {"checking"}, "credit_card": {"credit", "card"}}
