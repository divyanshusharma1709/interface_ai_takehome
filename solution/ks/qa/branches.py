"""Branch table: address, phone, hours, open-status and list answers from the locator and announcement pages."""
from __future__ import annotations

import re

from ..core.dates import month_day_year
from ..core.tokens import tokenize
from ..core.types import Result, SourceRef
from ..site_profile import (
    HOME_CITY,
    INNOVATION_LAB_WORDS,
    INSTITUTION,
    REGION_WORDS,
    STORE_BRANCH_WORD,
    is_home_url,
)
from .answer_type import BranchAsk, branch_ask
from .branch_build import (
    DAYS,
    STREET_WORDS,
    Branch,
    BranchBuilder,
    day_index,
    hours_on,
    render_schedule,
)

# ---- Questions about all branches at once: who is open on Saturday, are the hours the same, the usual hours.
_WEEKEND_WORD = re.compile(r"\b(saturday|sunday|weekend)s?\b")
_LIST_WORDS = re.compile(r"\bwhich\b|\bwhat\b|\blist\b|\bany\b")
_OPEN_WORD = re.compile(r"\bopen\b")
_SAME_OR_DIFFERENT = re.compile(r"\b(?:same|differ\w*|vary|varies|consistent|uniform)\b")
_TYPICAL_HOURS = re.compile(r"\bgeneral|typical|usual|standard|normal|regular|lobby|drive|branch hours|hours\b")
_DATE_SPECIFIC = re.compile(r"\bclos|open(?:ing)? date|still\b")
_DRIVE_UP = re.compile(r"drive[- ]?up")


def across(table, low: str, ask: BranchAsk) -> Result | None:
    weekend = _WEEKEND_WORD.search(low)
    if weekend and _LIST_WORDS.search(low) and _OPEN_WORD.search(low):
        return _open_on(table, weekend.group(1), low)
    if _SAME_OR_DIFFERENT.search(low) and ask.hours:
        return _compare(table, low)
    if ask.hours and _TYPICAL_HOURS.search(low) and not _DATE_SPECIFIC.search(low):
        return _typical(table, low)
    return None


def _schedule_kind(low: str) -> tuple[str, str]:
    """('drive', 'drive-up') when the question is about drive-up hours, else ('lobby', 'lobby')."""
    return ("drive", "drive-up") if _DRIVE_UP.search(low) else ("lobby", "lobby")


def _groups(table, attr: str):
    """Branches grouped by identical schedule, biggest group first."""
    groups: dict[tuple, list[Branch]] = {}
    for b in sorted(table.branches.values(), key=lambda b: b.key):
        sch = getattr(b, attr)
        if sch and b.status != "closed":
            groups.setdefault(tuple(sch), []).append(b)
    return sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0].key))


def _typical(table, low: str) -> Result | None:
    attr, label = _schedule_kind(low)
    groups = _groups(table, attr)
    if not groups:
        return None
    top, top_b = groups[0]
    names = ", ".join(b.name for b in top_b[:4])
    text = f"{label.capitalize()} hours vary by branch. "
    text += f"A typical schedule is {render_schedule(list(top))} ({names})."
    for sch, bs in groups[1:3]:
        text += f" Another schedule: {render_schedule(list(sch))} ({', '.join(b.name for b in bs[:3])})."
    text += " Which branch do you mean?"
    used = [b for _, bs in groups[:3] for b in bs[:2]]
    return Result(text, table.sources(used, 1) + table.general_sources(), True, None)


def _compare(table, low: str) -> Result | None:
    attr, label = _schedule_kind(low)
    groups = _groups(table, attr)
    if not groups:
        return None
    if len(groups) == 1:
        return Result(
            f"Yes, every branch with published {label} hours keeps {render_schedule(list(groups[0][0]))}.",
            table.sources(groups[0][1], 1),
            True,
            None,
        )
    text = f"No, {label} hours differ by branch. " + " ".join(
        f"{render_schedule(list(sch))} ({', '.join(b.name for b in bs[:4])})." for sch, bs in groups[:4]
    )
    return Result(
        text, table.sources([b for _, bs in groups[:4] for b in bs[:1]], 1) + table.general_sources(), True, None
    )


def _open_on(table, which: str, low: str) -> Result | None:
    day = 5 if which.startswith(("sat", "week")) else 6
    attr, label = _schedule_kind(low)
    open_b, closed_b = [], []
    for b in sorted(table.branches.values(), key=lambda b: b.key):
        sch = getattr(b, attr)
        if not sch or b.status == "closed":
            continue
        h = hours_on(sch, day)
        if h is None:
            continue
        (closed_b if h in ("closed", "ATM only") else open_b).append((b, h))
    if not open_b and not closed_b:
        return None
    dn = DAYS[day]
    text = (
        f"{label.capitalize()} open on {dn}: " + "; ".join(f"{b.name} ({h})" for b, h in open_b) + "."
        if open_b
        else f"No branch with published hours is open on {dn}."
    )
    if closed_b:
        text += f" Closed on {dn}: " + ", ".join(b.name for b, _ in closed_b) + "."
    text += f" Published hours cover {len(open_b) + len(closed_b)} of {len(table.branches)} known branches."
    return Result(text, table.sources([b for b, _ in open_b + closed_b], 1), True, None)

# words that name no branch when they are part of a label / street / alias
_NOT_A_NAME_IN_FIND = STREET_WORDS | {"learning", "lab", "point"}
_NOT_A_NAME_BY_ITSELF = STREET_WORDS | {
    "learning",
    "lab",
    "store",
    "first",
    "east",
    "west",
    "north",
    "south",
    "highway",
    "washington",
    "point",
    "business",
    "main",
    "central",
    "center",
    "pointe",
    "corridor",
}
# capitalised words in a question that are not another place
_COMMON_PLACE_WORDS = (
    {
        INSTITUTION.lower(),
        "road",
        "street",
        "avenue",
        "drive",
        "lane",
        "boulevard",
        "blvd",
        "rd",
        "st",
        "ave",
        "dr",
        "ln",
        "hwy",
        "highway",
        "branch",
        "interstate",
    }
    | {d.lower() for d in DAYS}
    | set(REGION_WORDS)
    | set(INNOVATION_LAB_WORDS)
)
_ATM_GENERIC = set(tokenize(f"{INSTITUTION} atm surcharge free branch store gas station")) | {"atm"}

_ATM_WORD = re.compile(r"\batms?\b")
_ATM_WHERE = re.compile(r"\bwhere\b|\blocat|\bnear\b|\bat\b")
_NOT_BRANCH_TOPIC = re.compile(
    r"\b(?:rate|apr|loan|card|account|atm|fee|dividend|member|call center|contact center|support|online|mobile)s?\b"
)
_PERSON_FACT = re.compile(r"\b(?:born|birth\w*|salary|age|manager|married|wife|husband|ceo|president)\b")
_COUNT_PLUS_DETAIL = (re.compile(r"\bhow many\b"), re.compile(r"\b(?:and|also|plus)\b"))
_TIME_OF_DAY = re.compile(
    r"\bwhat time\b|\buntil when\b|\bhow late\b|\bhow early\b|\bclos(?:e|es|ed|ing) (?:on|at|by|every|each)\b"
    r"|\bopen(?:s)? (?:on|until|till|at|late|early)\b"
    r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)days?\b|\bweekdays?\b|\bweekends?\b"
)
_NOT_TIME_OF_DAY = re.compile(
    r"\bwhich\b|\bhow many\b|\blast day\b|\bfinal day\b|\bpermanent\w*\b|\bclosing date\b|\bwhen did\b"
)
_WHICH_IS_CLOSING = (re.compile(r"\bwhich\b|\bwhat\b"), re.compile(r"clos(?:ing|e|ure)"))
_ANY_HOURS_WORD = re.compile(
    r"\bhours?\b|\bwhat time\b|\bwhen\b|\bopen(?:s|ing)?\b|\bclos\w+\b|\bsaturday\b|\bsunday\b|\bweekend"
)
_CLOSURE_ASK = re.compile(r"\bstill\b|\bclos(?:ing|ure)\b|\bshut|\bpermanent|\bclosed (?:down|for good)")
_FEATURES = (
    ("drive-up", r"drive[- ]?(?:up|thru|through)|drive[- ]?in"),
    ("video teller", r"video teller"),
    ("ATM", r"\batms?\b"),
    ("night deposit", r"night (?:deposit|drop)"),
    ("walk-up", r"walk[- ]?up"),
)
# branches listed in one answer
MAX_NAMED = 3
MAX_SOURCES = 6


class BranchTable:
    """Built once per snapshot view from `view.pages`."""

    def __init__(self, view):
        self.view = view
        built = BranchBuilder(view)
        self.branches: dict[str, Branch] = built.branches
        # (label, street, city, state, zip, page) of the surcharge-free ATMs the locator lists
        self.atms: list[tuple] = built.atms

    # ------------------------------------------------------------------ lookup
    def status_on(self, b: Branch, ref: str) -> str:
        if b.status == "closing" and b.status_date:
            return "closed" if ref >= b.status_date else "closing"
        return b.status

    def find(self, text: str) -> list[Branch]:
        """Branches named in `text` (street or label beats a bare city)."""
        q = set(tokenize(text))
        scored = []
        for b in self.branches.values():
            hit_name = q & b.name_tokens(_NOT_A_NAME_IN_FIND)
            hit_city = q & set(tokenize(b.city))
            if hit_name or hit_city:
                scored.append((len(hit_name) * 2 + len(hit_city), b))
        if not scored:
            return []
        top = max(s for s, _ in scored)
        return [
            b
            for s, b in sorted(scored, key=lambda t: (-t[0], t[1].key))
            if s == top and (top >= 2 or all(x[0] == top for x in scored))
        ]

    def known_places(self) -> set:
        out = set()
        for b in self.branches.values():
            out |= b.tokens()
        return out

    def unknown_name(self, raw: str) -> str | None:
        """A capitalised word in the question that is neither a known branch place nor an everyday word."""
        known = self.known_places() | _COMMON_PLACE_WORDS
        for i, w in enumerate(re.findall(r"[A-Za-z][A-Za-z'’-]+", raw)):
            if i == 0 or not w[0].isupper() or w.isupper() and len(w) <= 4 or w in ("I", INSTITUTION):
                continue
            if not set(tokenize(w)) & known and w.lower() not in known:
                return w
        return None

    def _names_branch(self, raw: str) -> bool:
        q = set(tokenize(raw))
        ignore = _NOT_A_NAME_BY_ITSELF
        for b in self.branches.values():
            if q & (b.name_tokens(ignore) - set(tokenize(STORE_BRANCH_WORD))):
                return True
            if b.city.lower() != HOME_CITY and set(tokenize(b.city)) & q:
                return True
        return False

    def sources(self, bs: list[Branch], n: int = 3) -> list[SourceRef]:
        out, seen = [], set()
        for b in bs:
            for url, sid, _date, quote in b.sources[-n:]:
                if url not in seen:
                    seen.add(url)
                    out.append(SourceRef(url, sid, quote))
        return out[:MAX_SOURCES]

    def general_sources(self) -> list[SourceRef]:
        out = []
        for page in sorted(self.view.pages.values(), key=lambda p: p.url):
            u = page.url.rstrip("/")
            if (u.endswith("/contact") or is_home_url(u)) and not page.parsed.listing:
                out.append(SourceRef(page.raw_url, page.snapshot_id, (page.title or "")[:200]))
        return out

    @staticmethod
    def address_of(b: Branch) -> str:
        parts = [b.street, b.city, f"{b.state} {b.zip}".strip()]
        return ", ".join(p for p in parts if p)

    # ------------------------------------------------------------------ entry point
    def answer(self, q, ref: str, as_of_given: bool, voice) -> Result | None:
        raw, low = q.raw, q.raw.lower()
        ask = branch_ask(low)
        if ask.explanation:
            # explanations come from the article, not from the table
            return None
        atm = self._atm_answer(raw, low, ask)
        if atm is not None:
            return atm
        about = ask.about_branch
        if not about and ask.any_detail and not _NOT_BRANCH_TOPIC.search(low):
            # 'the Waukee one', 'hours at the Hickman Road lab': a branch named without the word
            about = self._names_branch(raw)
        if not about or self.unknown_name(raw) or _PERSON_FACT.search(low):
            return None
        # LLM-extracted names only help find the branch the rules already routed here
        named = self.find(raw) or self.find(" ".join(q.entities))
        if named and all(rx.search(low) for rx in _COUNT_PLUS_DETAIL):
            # a count plus a detail about one branch: the whole passage answers both, not one table row
            return None
        if not named:
            return across(self, low, ask)
        return self._named_branches(named, low, ask, ref, as_of_given, voice)

    def _atm_answer(self, raw: str, low: str, ask: BranchAsk) -> Result | None:
        if self.atms and _ATM_WORD.search(low) and (ask.address or _ATM_WHERE.search(low)):
            return self._atm(raw)
        return None

    def _named_branches(
        self, named: list[Branch], low: str, ask: BranchAsk, ref: str, as_of_given: bool, voice
    ) -> Result | None:
        """Questions about one branch (or several in one city)."""
        asks_time = bool(_TIME_OF_DAY.search(low)) and not _NOT_TIME_OF_DAY.search(low)
        if asks_time and not any(b.lobby or b.drive for b in named):
            # 'listed as open' or a closing date does not answer a time-of-day question
            return voice.refusal("the opening hours of that branch are not published")
        if ask.status and all(rx.search(low) for rx in _WHICH_IS_CLOSING):
            movers = [b for b in named if b.status != "open"] or [
                b for b in self.branches.values() if b.status != "open"
            ]
            movers.sort(key=lambda b: (b.status_date, b.key), reverse=True)
            if movers:
                return self._status(movers[:MAX_NAMED], ref, as_of_given)
        feats = [name for name, rx in _FEATURES if re.search(rx, low)]
        if feats and not _ANY_HOURS_WORD.search(low):
            res = self._features(named, feats)
            if res is not None:
                return res
        res = self._detail(named, low, ask, ref, as_of_given)
        if res is None and len(named) == 1 and ask.any_detail:
            return voice.refusal(f"the {named[0].name} branch is known but the asked detail is not published")
        return res

    def _detail(self, named: list[Branch], low: str, ask: BranchAsk, ref: str, as_of_given: bool) -> Result | None:
        closure_q = bool(_CLOSURE_ASK.search(low)) or any(b.status != "open" for b in named)
        if ask.status and not closure_q and any(b.lobby or b.drive for b in named):
            # 'is X open Saturday' asks for the hours, not for 'listed as open'
            return self._hours(named, low)
        if ask.status:
            return self._status(named, ref, as_of_given)
        if ask.hours:
            res = self._hours(named, low)
            extra = (
                self._facts(named, ask.address, ask.phone, False)
                if (ask.phone or ask.address) and res is not None
                else None
            )
            # 'phone number and lobby hours': answer both parts
            if extra is not None:
                res = Result(f"{extra.text} {res.text}", res.sources, True, None)
            return res
        return self._facts(named, ask.address, ask.phone, ask.services)

    # ------------------------------------------------------------------ one branch
    def _atm(self, raw: str) -> Result | None:
        q = set(tokenize(raw))
        best = []
        for label, street, city, st, zp, page in self.atms:
            lt = set(tokenize(label)) - _ATM_GENERIC
            hit = lt & q
            if hit and len(hit) * 2 >= len(lt):
                best.append(
                    (
                        len(hit) + len(hit) / len(lt),
                        re.sub(r"(?i)\s*\batm\b\s*", " ", label).strip(),
                        street,
                        city,
                        st,
                        zp,
                        page,
                    )
                )
        if not best:
            return None
        top = max(b[0] for b in best)
        rows = [b for b in best if b[0] == top][:3]
        text = " ".join(
            f"The {r[1]} ATM (surcharge-free for {INSTITUTION} members) is at {r[2]}, {r[3]}, {r[4]} {r[5]}."
            for r in rows
        )
        return Result(
            text, [SourceRef(r[6].raw_url, r[6].snapshot_id, f"{r[1]} {r[2]}"[:200]) for r in rows[:1]], True, None
        )

    def _status(self, bs: list[Branch], ref: str, as_of_given: bool) -> Result | None:
        lines, used = [], []
        for b in bs[:MAX_NAMED]:
            st = self.status_on(b, ref)
            if b.status == "open" and not b.status_known and not b.sources:
                continue
            if st == "closed":
                lines.append(f"The {b.name} branch closed on {month_day_year(b.status_date)}.")
            elif st == "closing":
                lines.append(
                    f"The {b.name} branch is closing: this location will close {month_day_year(b.status_date)}."
                    if b.status_date
                    else f"The {b.name} branch is scheduled to close."
                )
            else:
                lines.append(f"The {b.name} branch is listed as open; no closure has been announced.")
            used.append(b)
        if not lines:
            return None
        return Result(" ".join(lines), self.sources(used), True, None)

    def _hours(self, bs: list[Branch], low: str) -> Result | None:
        want_drive = bool(re.search(r"drive[- ]?up|drive[- ]?through|drive thru", low))
        want_lobby = bool(re.search(r"\blobby\b|\binside\b", low))
        dm = re.search(r"\b(mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?s?\b", low)
        day = day_index(dm.group(1)) if dm else None
        weekend = bool(re.search(r"\bweekends?\b", low))
        parts, used = [], []
        for b in bs[:MAX_NAMED]:
            kinds = (
                [("drive-up", b.drive)]
                if want_drive and not want_lobby
                else (
                    [("lobby", b.lobby)]
                    if want_lobby and not want_drive
                    else [("lobby", b.lobby), ("drive-up", b.drive)]
                )
            )
            bits = self._hours_bits(kinds, day, weekend)
            if not bits:
                continue
            when = (
                f" on {DAYS[day]}"
                if day is not None and any(" hours not listed" not in x and ":" not in x.split(" ", 1)[0] for x in bits)
                else ""
            )
            parts.append(f"{b.name} hours{when} - " + "; ".join(bits) + ".")
            used.append(b)
        if not parts:
            return None
        return Result(" ".join(parts), self.sources(used), True, None)

    @staticmethod
    def _hours_bits(kinds: list, day: int | None, weekend: bool) -> list[str]:
        bits = []
        for label, sch in kinds:
            if not sch:
                continue
            if day is not None:
                w = hours_on(sch, day)
                bits.append(f"{label} {w if w else 'hours not listed'}" if w else f"{label}: {render_schedule(sch)}")
            elif weekend:
                w = [f"{d} {x}" for d, x in sch if any(day_index(t) in (5, 6) for t in d.split("–"))]
                bits.append(f"{label}: {'; '.join(w) if w else render_schedule(sch)}")
            else:
                bits.append(f"{label}: {render_schedule(sch)}")
        return bits

    def _features(self, bs: list, feats: list) -> Result | None:
        parts, used = [], []
        for b in bs[:MAX_NAMED]:
            if not b.services and not b.drive:
                continue
            listed = " ".join(b.services).lower() + (" drive-up" if b.drive else "")
            have = [f for f in feats if any(re.search(rx, listed) for n, rx in _FEATURES if n == f)]
            lack = [f for f in feats if f not in have]
            bits = ([f"has {', '.join(have)}"] if have else []) + ([f"has no {', no '.join(lack)}"] if lack else [])
            parts.append(
                f"The {b.name} branch "
                + " and ".join(bits)
                + (f" (services listed: {', '.join(b.services)})." if b.services else ".")
            )
            used.append(b)
        if not parts:
            return None
        return Result(" ".join(parts), self.sources(used), True, None)

    def _facts(self, bs, addr: bool, phone: bool, services: bool) -> Result | None:
        parts, used = [], []
        for b in bs[:MAX_NAMED]:
            bits = []
            if (addr or not (phone or services)) and b.street:
                bits.append(f"is at {self.address_of(b)}")
            if phone and b.phone:
                bits.append(f"phone number is {b.phone}")
            if services and b.services:
                bits.append("offers " + ", ".join(b.services))
            if bits:
                parts.append(f"The {b.name} branch " + "; ".join(bits) + ".")
                used.append(b)
        if not parts:
            return None
        return Result(" ".join(parts), self.sources(used), True, None)

