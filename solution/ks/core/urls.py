"""URL canonicalisation: page identity for the crawl and for source matching."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse

# Query parameters that never select a different page (crawl-side list; `utm_*` is dropped as well).
TRACKING_PARAMS = {
    "hslang",
    "campaignid",
    "scrollto",
    "containertype",
    "pageurl",
    "portalid",
    "webinteractivecontentid",
    "webinteractiveid",
    "hsctatracking",
    "gclid",
    "fbclid",
    "ref",
}


def _split(u: str):
    return urlparse(u if "//" in u else "https://" + u)


def normalize_url(u: str) -> str:
    """Canonical page id: lowercase host without `www.`, no fragment, no trailing slash."""
    p = _split(u.strip())
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    q = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    return f"https://{host}{path}" + (f"?{urlencode(q)}" if q else "")


def url_key(u: str) -> str:
    """Path-level comparison key (host + lowercase path, no query/fragment/trailing slash)."""
    p = _split(u.strip())
    return p.netloc.lower().removeprefix("www.") + p.path.rstrip("/").lower()






