"""Branch records, the parsing of opening-hours text ('Monday-Friday: 8:30 a.m.-5 p.m.') into schedules, and the branch records of one snapshot view."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.dates import MONTH_NUMBER
from ..core.tokens import tokenize
from ..site_profile import (
    INSTITUTION,
    LEARNING_LAB,
    LOCATOR_ATM_MARKER,
    LOCATOR_ATM_PREFIX,
    LOCATOR_BRANCH_MARKER,
    PLACE_MENTIONS,
    RELEASE_FOOTER,
    STATES,
    STORE_BRANCH_WORD,
    STORE_MENTION,
    PlaceMention,
    is_home_url,
    strip_title_suffix,
)

# ---- Branch records and the parsing of opening-hours text into schedules.
TIME_RX = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?|\bnoon\b|\bmidnight\b", re.I)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DAY_RX = r"(?:Mon|Tues?|Wed|Thu(?:rs)?|Fri|Sat|Sun)[a-z]*\.?"
_LINE = re.compile(rf"(?P<days>{_DAY_RX}(?:\s*(?:-|–|—|to|and|&)\s*{_DAY_RX})?)\s*:?\s*(?P<what>.+)$", re.I)
STREET_SUFFIX = (r"(?:Road|Rd\.?|Boulevard|Blvd\.?|Avenue|Ave\.?|Way|Street|St\.?|Drive|Dr\.?|Lane|Ln\.?|Parkway"
                 r"|Pkwy\.?|Court|Ct\.?|Highway|Hwy\.?)")
STREET_ADDRESS = re.compile(
    rf"(\d{{1,6}}\s+(?:[NSEW]\.?\s+)?[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){{0,3}}?\s+{STREET_SUFFIX})(?=[\s,.;)]|$)"
)
PHONE_AS_WRITTEN = re.compile(r"\(?\b\d{3}\)?[-. ]\s?\d{3}[-. ]\d{4}\b")

# street words and compass letters that say nothing about which place a name is
STREET_WORDS = frozenset(
    {
        "road",
        "rd",
        "street",
        "st",
        "avenue",
        "ave",
        "drive",
        "dr",
        "lane",
        "ln",
        "boulevard",
        "blvd",
        "way",
        "e",
        "w",
        "n",
        "s",
    }
)


def day_index(word: str) -> int | None:
    w = word.lower().strip(". ")
    for i, d in enumerate(DAYS):
        if d.lower().startswith(w[:3]):
            return i
    return None


def _clock(m: re.Match) -> tuple[int, int]:
    """(hour24, minute) of a matched time token."""
    t = m.group(0).lower()
    if t == "noon":
        return 12, 0
    if t == "midnight":
        return 0, 0
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    return (h % 12) + (12 if ap == "p" else 0), mi


def _format_clock(hm: tuple[int, int]) -> str:
    h, mi = hm
    if (h, mi) == (12, 0):
        return "noon"
    if (h, mi) == (0, 0):
        return "midnight"
    suffix = "a.m." if h < 12 else "p.m."
    h12 = h % 12 or 12
    return f"{h12}{':%02d' % mi if mi else ''} {suffix}"


def _hours_text(text: str) -> str:
    """'8:30 a.m.-5 p.m.' -> '8:30 a.m.–5 p.m.'; 'Closed' / 'ATM only' stay words; '' if unrecognised."""
    low = text.lower().strip(" .")
    if low.startswith("closed"):
        return "closed"
    if "atm only" in low:
        return "ATM only"
    times = [_clock(m) for m in TIME_RX.finditer(text)]
    if len(times) >= 2:
        return f"{_format_clock(times[0])}–{_format_clock(times[1])}"
    return ""


def parse_schedule(lines: list[str]) -> list[tuple[str, str]]:
    """[('Monday–Friday', '8:30 a.m.–5 p.m.'), ('Sunday', 'closed')] from 'Day-range: hours' lines."""
    out = []
    for ln in lines:
        m = _LINE.match(ln.strip(" -*•\t"))
        if not m:
            continue
        what = _hours_text(m.group("what"))
        if not what:
            continue
        parts = re.split(r"\s*(?:-|–|—|\bto\b)\s*", m.group("days").strip(), maxsplit=1)
        idx = [day_index(x) for x in parts if x]
        if any(i is None for i in idx) or not idx:
            continue
        if len(idx) == 2 and idx[0] != idx[1]:
            days = f"{DAYS[idx[0]]}–{DAYS[idx[1]]}"
        else:
            days = DAYS[idx[0]]
        out.append((days, what))
    return out


def hours_on(schedule: list[tuple[str, str]], day: int) -> str | None:
    """The hours text of `schedule` on weekday index `day` (0 = Monday)."""
    for days, what in schedule:
        a, _, b = days.partition("–")
        lo, hi = day_index(a), day_index(b or a)
        if lo is not None and hi is not None and lo <= day <= hi:
            return what
    return None


def render_schedule(schedule: list[tuple[str, str]]) -> str:
    return "; ".join(f"{d} {w}" for d, w in schedule)


@dataclass
class Branch:
    key: str
    city: str = ""
    state: str = ""
    # '1090 E. Hickman Road'
    street: str = ""
    zip: str = ""
    phone: str = ""
    # the name the site gives ('East Washington')
    label: str = ""
    lobby: list = field(default_factory=list)
    drive: list = field(default_factory=list)
    services: list = field(default_factory=list)
    # open | closing | closed
    status: str = "open"
    # ISO date the closure takes effect
    status_date: str = ""
    # ISO date of the post announcing it
    status_known: str = ""
    # (url, snapshot_id, date, quote)
    sources: list = field(default_factory=list)
    # other names: the intersection or area a post gives ('Highway 100 and Edgewood Road')
    aliases: list = field(default_factory=list)
    # branch | innovation lab | operations center
    kind: str = "branch"

    @property
    def name(self) -> str:
        if self.label and self.city and self.label.lower() != self.city.lower():
            return f"{self.city} ({self.label})"
        return self.label or self.city or self.street

    @property
    def street_name(self) -> str:
        """The street without its number."""
        return re.sub(r"^\d+\s+", "", self.street)

    def tokens(self) -> set:
        s = set(tokenize(" ".join([self.label, self.city, self.street_name] + self.aliases)))
        return s - STREET_WORDS - {"branch"}

    def name_tokens(self, ignore: frozenset | set) -> set:
        """Tokens of the label, street and aliases (not the city) without the words in `ignore`."""
        return set(tokenize(" ".join([self.label, self.street_name] + self.aliases))) - ignore

# ---- Builds the branch records of one snapshot view from its pages.
_LOCATOR_ENTRY = re.compile(
    rf"(?m)^{re.escape(LOCATOR_BRANCH_MARKER)}\s*\n+\s*(\d{{1,6}} [^,\n]+), ([A-Za-z .'-]+), ([A-Z]{{2}}),? (\d{{5}}"
    rf")\s*\n+([^\n]*)\n+([^\n]*)"
)
_LOCATOR_ATM = re.compile(
    rf"(?m)^###\s+([^\n]+)\n+{re.escape(LOCATOR_ATM_MARKER)}\s*\n+(\d{{1,6}} [^,\n]+), ([^,\n]+), ([A-Z]{{2}}),? "
    rf"(\d{{5}})"
)
_DATELINE = re.compile(r"(?m)^([A-Z][A-Z .'-]{2,}),\s*([A-Z][A-Za-z.]+)\s*[–—-]")
_ANNOUNCED_CITY = re.compile(rf"The (\w+) {re.escape(INSTITUTION)} branch is located")
_LAB = re.compile(re.escape(LEARNING_LAB), re.I)
_AREA_NOISE = {
    "the",
    "and",
    "of",
    "at",
    "highway",
    "hwy",
    "road",
    "street",
    "avenue",
    "interstate",
    "intersection",
    "corner",
    "just",
    "off",
    "in",
    "near",
    "next",
}
_STREET_TAIL = re.compile(r"\s+(?:Rd|Road|Dr|Drive|St|Street|Ave|Avenue|Ln|Lane|Blvd)\.?$")


def _quote_key(city: str, street: str, fallback: str) -> str:
    return f"{city.lower()}|{re.sub(r'^[0-9]+ ', '', street).lower() or fallback.lower()}"


class BranchBuilder:
    """Fills `branches` and `atms` from the locator, announcement and closure pages."""

    def __init__(self, view):
        self.branches: dict[str, Branch] = {}
        self.atms: list[tuple] = []
        pages = sorted(view.pages.values(), key=lambda p: (p.publish_date or "", p.url))
        for page in pages:
            if page.url.rstrip("/").endswith("/contact") or page.page_type == "branch_contact":
                text = page.parsed.text or ""
                self._from_locator(page, text)
                self._atms_from_locator(page, text)
        for page in pages:
            if page.page_type == "newsroom":
                self._from_newsroom(page)
        # places the posts mention without a branch table of their own
        for page in pages:
            if page.page_type in ("newsroom", "blog") or is_home_url(page.url):
                self._from_mentions(page, page.parsed.text or "")

    def _get(self, key: str) -> Branch:
        return self.branches.setdefault(key, Branch(key))

    @staticmethod
    def _src(page, quote: str) -> tuple:
        return (page.raw_url, page.snapshot_id, page.publish_date or "", quote[:200])

    def _from_newsroom(self, page) -> None:
        text = page.parsed.text or ""
        title = strip_title_suffix(page.title or "")
        if re.search(r"\b(?:close|closing|closure)\b", title, re.I) and re.search(
            r"\bbranch|location\b", title + text[:600], re.I
        ):
            self._from_closure(page, text, title)
        elif re.search(r"Branch hours|located at \d|phone number for", text) and re.search(
            r"[Bb]ranch|[Ll]earning [Ll]ab", text[:1500]
        ):
            self._from_announcement(page, text)

    # ------------------------------------------------------------------ locator list
    def _from_locator(self, page, text: str) -> None:
        for m in _LOCATOR_ENTRY.finditer(text):
            street, city, st, zp = m.group(1).strip(), m.group(2).strip(), m.group(3), m.group(4)
            services = [x.strip() for x in re.split(r"·|•", m.group(6)) if x.strip()] if "·" in m.group(6) else []
            b = self._get(f"{city.lower()}|{re.sub(r'^[0-9]+ ', '', street).lower()}")
            b.city, b.state, b.street, b.zip = city, st, street, zp
            b.label = b.label or _STREET_TAIL.sub("", re.sub(r"^\d+\s+", "", street))
            b.services = services or b.services
            b.sources.append(self._src(page, m.group(0).replace("\n", " ")))

    def _atms_from_locator(self, page, text: str) -> None:
        for m in _LOCATOR_ATM.finditer(text):
            label = re.sub(rf"^{re.escape(LOCATOR_ATM_PREFIX)}", "", m.group(1)).strip()
            self.atms.append((label, m.group(2).strip(), m.group(3).strip().title(), m.group(4), m.group(5), page))

    # ------------------------------------------------------------------ announcements
    @staticmethod
    def _dateline(text: str) -> tuple[str, str]:
        m = _DATELINE.search(text)
        if m:
            return m.group(1).title().strip(), STATES.get(m.group(2), m.group(2))
        return "", ""

    def _from_announcement(self, page, text: str) -> None:
        city, state = self._dateline(text)
        if not city:
            m = _ANNOUNCED_CITY.search(text)
            city = m.group(1) if m else ""
        located = text.find("located at")
        addr = STREET_ADDRESS.search(text[located:]) if "located at" in text else None
        street = addr.group(1).strip() if addr else ""
        pm = re.search(r"phone number for the ([\w .-]+?) (?:branch )?is", text)
        label = pm.group(1).strip() if pm else (LEARNING_LAB if LEARNING_LAB in text else "")
        b = self._get(_quote_key(city, street, label))
        b.city, b.state = city or b.city, state or b.state
        b.street = street or b.street
        self._add_area_alias(b, text, located)
        if _LAB.search(text[:1800]):
            b.aliases = sorted(set(b.aliases) | {LEARNING_LAB})
        b.label = _LAB.sub(LEARNING_LAB, label) or b.label
        ph = PHONE_AS_WRITTEN.search(text[pm.start():pm.start() + 200]) if pm else \
            PHONE_AS_WRITTEN.search(text[text.find("phone number"):][:200]) if "phone number" in text else None
        if ph:
            # quoted exactly as the page writes it, e.g. '(608) 282-6000'
            b.phone = ph.group(0).strip()
        self._add_announced_hours(b, text)
        if re.search(r"video teller", text, re.I) and "video teller" not in b.services:
            b.services.append("video teller")
        if b.drive and "drive-up" not in b.services:
            b.services.append("drive-up")
        b.sources.append(self._src(page, (street + " " + b.phone).strip() or page.title or ""))
        b.status_known = b.status_known or (page.publish_date or "")

    @staticmethod
    def _add_area_alias(b: Branch, text: str, located: int) -> None:
        """'at the intersection of Highway 100 and Edgewood Road' names the branch by its area."""
        am = (
            re.search(
                r"(?:intersection of|corner of|just off)\s+([^,.]+?)(?:\s+in\s+[A-Z]|[,.]|$)",
                text[located : located + 300],
            )
            if "located at" in text
            else None
        )
        if am:
            area = " ".join(w for w in am.group(1).split() if w.lower() not in _AREA_NOISE and not w.isdigit())
            if area:
                b.aliases = sorted(set(b.aliases) | {area})

    @staticmethod
    def _add_announced_hours(b: Branch, text: str) -> None:
        hi = text.find("Branch hours")
        if hi < 0:
            hi = text.find("Lobby")
        if hi < 0:
            return
        block = re.split(re.escape(RELEASE_FOOTER), text[hi:hi + 700])[0]
        parts = re.split(r"(?i)(?=drive-?up)", block, maxsplit=1)
        lobby_part, drive_part = parts[0], (parts + [""])[1]
        lobby_lines = [re.sub(r"(?i)^(?:branch hours are:\s*)?lobby\s*", "", x.strip()) for x in lobby_part.split("\n")]
        drive_lines = [re.sub(r"(?i)^drive-?up\s*", "", x) for x in drive_part.split("\n")]
        b.lobby = parse_schedule(lobby_lines) or b.lobby
        b.drive = parse_schedule(drive_lines) or b.drive

    # ------------------------------------------------------------------ prose mentions
    def _from_mentions(self, page, text: str) -> None:
        for place in PLACE_MENTIONS:
            matches = (
                [m for m in [place.pattern.search(text)] if m]
                if place.first_only
                else list(place.pattern.finditer(text))
            )
            for m in matches:
                self._add_mention(place, m, page)
        store = STORE_MENTION.pattern.search(text)
        if store:
            for b in self.branches.values():
                if STORE_BRANCH_WORD.lower() in b.key or b.label.lower() == STORE_BRANCH_WORD.lower():
                    b.street = b.street or STORE_MENTION.street_template.format(store.group(1))
                    b.aliases = sorted(set(b.aliases) | {STORE_MENTION.alias})

    def _add_mention(self, place: PlaceMention, m: re.Match, page) -> None:
        city = place.city or m.group(2)
        b = self._get(f"{city.lower()}|{place.label.lower()}")
        b.city, b.state, b.street, b.label, b.kind = (
            city,
            place.state,
            place.street or m.group(1).strip(),
            place.label,
            place.kind,
        )
        if place.services:
            b.services = b.services or list(place.services)
        if place.aliases:
            b.aliases = sorted(set(b.aliases) | set(place.aliases))
        b.sources.append(self._src(page, m.group(0)))

    # ------------------------------------------------------------------ closures
    def _from_closure(self, page, text: str, title: str) -> None:
        pub = page.publish_date or ""
        nm = re.search(r"\b([A-Z][\w.-]+(?: [A-Z][\w.-]+)?) branch\b", title) or re.search(
            r"\b([A-Z][\w.-]+) (?:location|branch)\b", text
        )
        label = nm.group(1) if nm else ""
        iso = self._closing_date(text, pub)
        am = STREET_ADDRESS.search(text)
        city_m = re.search(r"(\d{1,6} [^.]*?) in ([A-Z][a-z]+), ", text)
        street = am.group(1) if am else ""
        city = (
            city_m.group(2)
            if (city_m and label.lower() in text[max(0, city_m.start() - 60) : city_m.end()].lower())
            else ""
        )
        if (
            not city
            and label
            and re.search(rf"{re.escape(label)}\b", text)
            and label.lower() != STORE_BRANCH_WORD.lower()
        ):
            city = label
        key = _quote_key(city or label, street, label)
        existing = next((b for b in self.branches.values() if label and label.lower() in b.tokens()), None)
        b = existing or self._get(key)
        b.label = b.label or label
        b.city = b.city or city
        b.street = b.street or street
        b.status = "closing"
        b.status_date = iso or b.status_date
        b.status_known = pub
        m = re.search(r"[^.]*(?:will close|last day of service)[^.]*\.", text)
        b.sources.append(self._src(page, m.group(0).strip() if m else title))

    @staticmethod
    def _closing_date(text: str, published: str) -> str:
        """ISO date of 'will close on March 31' (the post's year, or the next)."""
        dm = re.search(
            r"(?:last day of service will be|will close|closing on"
            r"|closes?)\s+(?:on\s+)?(?:\w+day,\s+)?([A-Z][a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
            text,
        )
        if not (dm and dm.group(1).lower()[:3] in MONTH_NUMBER):
            return ""
        mon = MONTH_NUMBER[dm.group(1).lower()[:3]]
        year = (
            int(dm.group(3))
            if dm.group(3)
            else (int(published[:4]) + (1 if published and mon < int(published[5:7]) else 0) if published else 0)
        )
        return f"{year:04d}-{mon:02d}-{int(dm.group(2)):02d}" if year else ""

