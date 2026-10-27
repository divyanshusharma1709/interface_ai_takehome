"""Figures a site states more than once with different numbers: matched by the words around the figure."""
# Winner: the newest dated statement (for a phone number, the number most pages print); a dated official statistics box
# overrides press-release boilerplate when it is at least as new.
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..core.dates import MONTH_WORD, days_between
from ..core.types import SourceRef
from ..ingest.site_stats import best_box, box_wins
from . import vocab as V
from .base import EDITORIAL, QUOTE_MAX, STOP_WORDS, Cand, Miner, authority_of, path_of, strip_origin

# their numbers are the fact store's business; only phones are read there
TABULAR_PAGE_TYPES = ("rate_sheet", "product")
BOILERPLATE_PAGES = 3              # a value repeated on this many pages is template text
# statements older than this (against the newest dated release) are a historical series
STALE_RELEASE_DAYS = 730
RESTATEMENT_DAYS = 300             # statements further apart than this must be near-identical sentences ...
RESTATEMENT_SIMILARITY = 0.85      # ... to be a re-statement and not a different year's story

_SCALED = re.compile(r"million|billion|thousand|[$%]")
_QUANTIFIER = (r"(?:(?:more than|over|nearly|almost|about|approximately|around|at least|up to|just over|under|"
               r"fewer than|roughly)\s+)?")
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
# Date and time slots are matched only so their digits are not read as figures.
SLOT = re.compile(
    r"(?P<date>\b" + MONTH_WORD + r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b|\b" + MONTH_WORD
    + r"\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b)"
    r"|(?P<time>\b\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?)(?!\w))"
    r"|(?P<phone>(?<![\d.])(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d))"
    r"|(?P<ord>\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth"
    r"|\d+(?:st|nd|rd|th))\b)"
    r"|(?P<num>" + _QUANTIFIER + r"\$?" + _NUMBER + r"(?:\s?(?:%|percent|million|billion|thousand)\b|%)?)", re.I)


@dataclass
class Occurrence:
    # the value, comparable across spellings: ("num", unit, amount) | ("phone", digits) | ("ord", word)
    canon: tuple
    raw: str
    phrase: str                # the value as displayed: the figure plus the noun or ordinal cue that follows it
    stem: str                  # the words around the figure, 'N' standing for it
    tokens: frozenset          # every word of the sentence
    url: str
    pv: object
    date: str | None


def canonical(kind, raw):
    if kind == "num":
        v = float(re.search(r"\$?\s*(\d[\d,]*(?:\.\d+)?)", raw).group(1).replace(",", ""))
        s = raw.lower()
        v *= 1e9 if "billion" in s else 1e6 if "million" in s else 1e3 if "thousand" in s else 1
        return ("num", "$" if "$" in raw else "%" if "%" in s or "percent" in s else "", round(v, 4))
    return ("phone", re.sub(r"\D", "", raw)[-10:]) if kind == "phone" else ("ord", raw.lower())


def is_figure(raw, words, pos, after):
    # A number worth comparing: not a bound, a year, an identifier, a street number or a list index.
    digits = re.sub(r"\D", "", raw)
    prev = words[pos - 1] if pos else ""
    if raw.strip().lower().startswith(("under", "fewer")) or prev in V.NUMBER_SKIP_BEFORE or prev == "#phone":
        return False
    if re.fullmatch(r"(?:19|20)\d\d", raw.strip()) or len(digits) >= 9 or (len(digits) == 5 and "," not in raw):
        return False
    if (raw.strip().startswith("0") and "." not in raw) or (re.fullmatch(r"\s*\d+\s*", raw) and int(digits) < 10):
        return False
    return not any(V.STREET_WORD.match(a) for a in after[:2])


def occurrences_in(sent, url, pv):
    # (group key, Occurrence) for each comparable figure of a sentence; the group key is the words around the
    # figure.
    # worked payment examples are illustrations, not a statistic
    if V.PAYMENT_EXAMPLE.search(sent):
        return
    toks, slots, last = [], [], 0                      # toks: (word, None) | ('#kind', slot index)
    for m in SLOT.finditer(sent):
        toks += [(w, None) for w in re.findall(r"[a-z]+", sent[last:m.start()].lower())]
        slots.append((m.lastgroup, m.group(0)))
        toks.append((f"#{m.lastgroup}", len(slots) - 1))
        last = m.end()
    toks += [(w, None) for w in re.findall(r"[a-z]+", sent[last:].lower())]
    words = [t for t, _ in toks]
    for pos, (_, si) in enumerate(toks):
        if si is None or slots[si][0] in ("time", "date"):
            continue
        kind, raw = slots[si]
        before = words[max(0, pos - (4 if kind == "phone" else 2)):pos]
        after = [] if kind == "phone" else words[pos + 1:pos + 4]
        if (kind == "num" and not is_figure(raw, words, pos, after)) or \
                len([w for w in before + after if w not in STOP_WORDS and not w.startswith("#")]) < 2:
            continue
        canon = canonical(kind, raw)
        cue = [a for a in after[:2] if a in V.ORDINAL_CUES] if kind == "ord" else []
        noun = ""                                      # the noun right after a number that says what it counts
        if kind == "num" and after and not after[0].startswith("#"):
            if after[0] in V.BRANCH_WORDS:
                noun = " ".join(w for w in after[:3] if w not in STOP_WORDS)
            elif after[0].endswith("s") and after[0] not in STOP_WORDS:
                noun = after[0]
        key = (kind, tuple(before), tuple(after))
        phrase = " ".join(x for x in [raw.strip().rstrip("."), " ".join(cue) if cue else noun] if x)
        if kind == "num" and noun in V.NOUN_GROUP and canon[2] >= 100 and not _SCALED.search(raw):
            key = ("noun", noun, canon[1])
        if kind == "phone":
            dashed = "-".join([canon[1][:3], canon[1][3:6], canon[1][6:]])
            phrase = dashed if dashed == raw.strip() else f"{dashed} ({raw.strip()})"
        date = (pv.publish_date
            or (None if pv.page_type in EDITORIAL else pv.effective_date or (pv.fetched_at or pv.taken_on)[:10]))
        yield key, Occurrence(canon, raw, phrase, " ".join(before + ["N"] + after),
                              frozenset(w for w in words if not w.startswith("#")), url, pv, date)


def statistic_name(key, occ, kind):
    # Which site-level statistic a cluster is about (members / assets / employees / branches), or None.
    if kind != "num":
        return None
    if key[0] == "noun":
        return key[1] if key[1] in V.COUNTED_NOUNS else None
    words = set().union(*(o.tokens for o in occ))
    around = set(key[1]) | set(key[2])
    unit = occ[0].canon[1]
    if unit == "$" and V.ASSETS_WORD in around:
        return "assets"
    if (unit == "" and (V.MEMBERS_CUES & words) and V.ASSETS_WORD in around | words
        and not (set(V.BRANCH_WORDS) & around)):
        return "members"
    return "branches" if set(V.BRANCH_WORDS) & around else None


def _score(kind, boiler, ordinal, occ, urls):
    # (confidence, salience) of a standing claim; repeated template text, ranking streaks, institutional nouns and
    # phones make it clearer
    state_noun = bool(set().union(*(o.tokens for o in occ)) & V.STATE_NOUNS)
    conf = 0.45 + 0.25 * boiler + 0.2 * ordinal + 0.1 * state_noun + (0.1 if kind == "phone" else 0)
    if not boiler and not ordinal and kind == "num":
        conf -= 0.05                                   # a one-off figure with no standing-claim cue
    conf = min(conf, 0.9)
    sal = 0.95 if (state_noun or ordinal) else 0.7     # the institution's own statistics matter most to a member
    if kind == "num" and not boiler:                   # a program total or limit stated two ways
        conf += 0.2
        sal = 0.85
    return conf, sal * (1 + 0.03 * min(len(urls), 10))


def _official_box(box, name, newest, values, reason, srcs):
    # A dated official statistics box that is at least as new as the newest statement outranks the boilerplate.
    b, label, value = box
    top = value + (" members" if name == "members" else "")
    values = ([top] + [v for v in values if re.sub(r"[^0-9.]", "", v) != re.sub(r"[^0-9.]", "", value)])[:5]
    reason = (f"The dated official statistics box on {path_of(b.url)} (As of: {b.date_text}) states {label}: {value}; a "
              f"dated box outranks press-release boilerplate when its as-of date is at least as new as the newest "
              f"statement ({newest or 'undated'}). " + reason)
    srcs = [SourceRef(b.raw_url, b.snapshot_id, f"{b.line} | {label}: {value}")] + srcs[:2]
    return values, top, reason, srcs


class StatMiner(Miner):
    def __init__(self, kb):
        super().__init__(kb)
        self._page_digits = None

    def run(self):
        if not self.kb.views:
            return []
        groups = defaultdict(list)
        for url, pv in self.pages(self.kb.views[-1], skip_editorial=False, html_only=True, skip_calculators=True):
            tabular, seen = pv.page_type in TABULAR_PAGE_TYPES, set()
            for sent in self.sentences(pv.parsed.text, 25, 400, ("|", "#"), True):
                for g, occ in occurrences_in(sent, url, pv):
                    if (not tabular or g[0] == "phone") and (g, occ.canon) not in seen:
                        seen.add((g, occ.canon))
                        groups[g].append(occ)
        latest = max((o.pv.publish_date for g in groups.values() for o in g if o.pv.publish_date), default=None)
        found = []
        for key in sorted(groups):
            for cluster in ([groups[key]] if key[0] == "noun" else self._split(groups[key])):
                got = self._candidate(key, cluster, latest)
                if got:
                    found.append(got)
        return self._merge(found)

    @staticmethod
    def _split(occs):
        # Separate statements that merely share a window: greedy clusters by sentence-word overlap.
        clusters = []
        for o in sorted(occs, key=lambda o: o.url):
            best, best_sim = None, 0.25
            for c in clusters:
                sim = len(o.tokens & c[0].tokens) / max(1, len(o.tokens | c[0].tokens))
                if sim >= best_sim:
                    best, best_sim = c, sim
            if best is None:
                clusters.append([o])
            else:
                best.append(o)
        return clusters

    def _pages_printing(self, digits):
        # How many pages of the newest crawl print this phone number.
        if self._page_digits is None:
            self._page_digits = [re.sub(r"\D", "", re.sub(r"\d{4}-\d{2}-\d{2}", "",
                pv.parsed.text)) for pv in self.kb.views[-1].pages.values()]
        return sum(digits in d for d in self._page_digits)

    @staticmethod
    def _is_standing_claim(kind, key, occ, by_val, n_top, latest):
        boiler = n_top[0] >= BOILERPLATE_PAGES
        if kind == "ord":      # 'Nth year of <program>' marks an event; only 'the fourth time' style streaks are claims
            return any(w in V.ORDINAL_CUES for w in key[2][:2]) and any(w in V.ORDINAL_CLAIM_WORDS for w in key[2][:3])
        if kind == "phone":
            return len(n_top) >= 2 and n_top[0] >= 2
        if len({c[1] for c in by_val}) > 1:
            return False                                    # '$10' vs '20%': different things, not one figure
        if (key[0] != "noun"
            and any(w.endswith("ed") and len(w) > 4 and w not in V.STATIVE_PAST_WORDS for o in occ for w in o.tokens)):
            return False                                    # past-tense event figures differ by year, not by error
        if not boiler:
            unit, magnitude = next(iter(by_val))[1:]
            up_to = all(o.raw.strip().lower().startswith("up to") for o in occ)
            if not ((unit == "$" and magnitude >= 1e6) or (unit == "%" and up_to)):
                return False                                # one-off figures need to be program totals / limits
        dates = sorted(o.date for o in occ if o.date)
        if dates and latest and days_between(dates[-1], latest) > STALE_RELEASE_DAYS:
            return False                                    # historical series of old releases
        if not boiler and len(dates) >= 2 and days_between(dates[0], dates[-1]) > RESTATEMENT_DAYS:
            sets = [o.tokens for o in occ]
            # else: different years' stories
            return min(len(a & b) / len(a | b) for a in sets for b in sets) >= RESTATEMENT_SIMILARITY
        return True

    def _candidate(self, key, occ, latest):
        # (candidate, the cluster's figures as text) or None when the cluster is not a contradiction worth
        # reporting.
        kind = "num" if key[0] == "noun" else key[0]
        by_val = defaultdict(list)
        for o in occ:
            by_val[o.canon].append(o)
        urls = {o.url for o in occ}
        n_top = sorted((len({o.url for o in v}) for v in by_val.values()), reverse=True)
        if len(by_val) < 2 or len(urls) < 2 or not self._is_standing_claim(kind, key, occ, by_val, n_top, latest):
            return None
        boiler, ordinal = n_top[0] >= BOILERPLATE_PAGES, kind == "ord"
        if kind == "phone":                    # the number most pages use is the institution's; then the official page
            pages = {c: self._pages_printing(c[1]) for c in by_val}
            rank = lambda o: (pages[o.canon], authority_of(o.pv), o.date or "", o.url)
        else:
            rank = lambda o: (o.date or "", authority_of(o.pv), o.url)
        win = max(occ, key=rank)
        loser = max((o for o in occ if o.canon != win.canon), key=rank)
        conf, sal = _score(kind, boiler, ordinal, occ, urls)
        values = self._values(win, by_val)
        forms = Counter(o.phrase for o in by_val[win.canon]
                        if o.phrase != values[0] and o.phrase.split()[-1] == values[0].split()[-1])
        resolution = values[0] + (" (" + "; ".join(f for f, _ in forms.most_common(2)) + ")" if forms else "")
        rule = ("The number printed by the most pages wins (then the most authoritative page, "
                "then newest)" if kind == "phone"
                else "Newest dated statement wins")
        reason = (f"{rule}: {win.phrase!r} on {strip_origin(win.url)} ({win.date or 'undated'}) vs {loser.phrase!r} on "
                  f"{strip_origin(loser.url)} ({loser.date or 'undated'}); {len(by_val)} different values across "
                  f"{len(urls)} pages "
                  f"({'repeated boilerplate' if boiler else 'series of releases'}); older pages keep the figure that "
                  f"was true when written.")
        srcs = [SourceRef(o.pv.raw_url, o.pv.snapshot_id, o.stem[:QUOTE_MAX]) for o in (win, loser)]
        name = statistic_name(key, occ, kind)
        topic = (V.STAT_TOPICS.get(name or "")
            or (V.COUNTIES_TOPIC if key[0] == "noun" and key[1] == "counties" else self._topic(key, occ, kind)))
        box = best_box(self.kb.views[-1], name) if name else None
        newest = max((o.date for o in occ if o.date), default=None)
        if box and box_wins(box[0], newest):
            values, resolution, reason, srcs = _official_box(box, name, newest, values, reason, srcs)
            conf = min(conf + 0.05, 0.99)
        return Cand("stat", topic, values, resolution, reason, srcs, round(conf, 3), round(sal, 3),
            f"stat:{key}:{sorted(by_val)[0]}"), \
            sorted(map(str, by_val))

    @staticmethod
    def _values(win, by_val):
        # The winner's spelling first, then the most recent and the most widely printed other values (at
        # most five).
        newest_of = lambda c: max((o.date or "") for o in by_val[c])
        rest = sorted((c for c in by_val if c != win.canon), key=lambda c: (newest_of(c), str(c)), reverse=True)
        common = sorted(rest, key=lambda c: (-len({o.url for o in by_val[c]}), str(c)))
        values = []
        for c in list(dict.fromkeys([win.canon] + rest[:2] + common[:2] + rest))[:5]:
            forms = Counter(x.phrase for x in by_val[c])
            ph = max(forms, key=lambda f: (forms[f], f))
            for rx, alias in V.STAT_ALIASES:
                m = rx.search(ph.lower())
                if m and alias not in ph.lower():
                    ph = ph + f" ({ph.lower()[:m.start()].strip()} {alias})".replace("( ", "(")
            if ph not in values:
                values.append(ph)
        return values

    @staticmethod
    def _topic(key, occ, kind):
        # A noun phrase from the words around the figure plus the state nouns every occurrence shares.
        words = ([key[1]] if key[0] == "noun"
            else [w for w in list(key[1]) + list(key[2]) if w not in STOP_WORDS and not w.startswith("#")])
        for w in sorted(frozenset.intersection(*(o.tokens for o in occ)) & V.STATE_NOUNS):
            if w not in words:
                words.append(w)
        text = " ".join(dict.fromkeys(words)) or "figure"
        return text[:1].upper() + text[1:] + {"ord": " (ordinal recognition)", "phone": " (phone number)"}.get(kind, "")

    @staticmethod
    def _merge(found):
        # Templates that quote the same figures (membership counts in three boilerplate variants) are one
        # topic.
        merged = []                                        # [candidate, its figures]
        for cand, figures in sorted(found, key=lambda f: (-f[0].score, f[0].key)):
            cv = set(figures)
            for m in merged:
                shared = len(cv & m[1])
                if shared >= 2 and shared / len(cv | m[1]) >= 0.4:
                    for v in cand.values:
                        if v not in m[0].values and len(m[0].values) < 8:
                            m[0].values.append(v)
                    m[1] |= cv
                    break
            else:
                merged.append([cand, cv])
        return [m[0] for m in merged]
