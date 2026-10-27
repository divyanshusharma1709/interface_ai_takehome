"""Ranking (page authority, effective date, topical fit) and wording of facts for answers."""

from __future__ import annotations

import re

from .facts import AUTHORITY, Fact

# ---- fact_rank
_WEAK_URL = re.compile(
    r"-lp(?:$|/)|/offer|calculator|/learn/|/newsroom|/blog|/resources|checking-offer|makethemost|"
    r"newcard|homebuying-guidance|mortgage-refi",
    re.I,
)
_FAMILY_URL = {
    "certificate": ("savings-account", "certificate", "rates"),
    "savings": ("savings", "rates"),
    "money_market": ("savings", "money-market", "moneymarket", "rates"),
    "checking": ("checking", "rates"),
    "ira": ("retirement", "savings-account", "rates"),
    "hsa": ("savings", "rates"),
    "credit_card": ("credit-cards", "rates"),
    "mortgage": ("home-loans", "rates"),
    "auto_loan": ("auto", "rates"),
    "personal_loan": ("personal-loans", "rates"),
    "heloc": ("home-equity", "rates"),
    "home_equity_loan": ("home-equity", "rates"),
    "home_improvement_loan": ("home-improvement", "home-equity", "rates"),
    "recreational_loan": ("recreational", "rates"),
    "fee": ("fee-schedule",),
}


def url_fit(url: str, family: str) -> int:
    path = url.lower()
    return sum(1 for w in _FAMILY_URL.get(family, ()) if w in path) * 2 + (
        1 if path.rstrip("/").endswith("rates") else 0
    )


def authority(f: Fact) -> float:
    return AUTHORITY.get(f.page_type, 1.0) - (1.5 if _WEAK_URL.search(f.url) else 0.0)


def rank_key(f: Fact):
    """Which observation of the same key wins: authority, then effective date, then topical fit, then crawl time."""
    return (f.confidence >= 0.6, authority(f), f.effective_date or "", url_fit(f.url, f.family), f.fetched_at, f.url)


# ---- fact_format
LOAN_FAMILIES_EXCL_CARD = {
    "mortgage",
    "auto_loan",
    "personal_loan",
    "heloc",
    "home_equity_loan",
    "home_improvement_loan",
    "recreational_loan",
}
_ACRONYM = {
    "ira": "IRA",
    "atv": "ATV",
    "hsa": "HSA",
    "heloc": "HELOC",
    "rv": "RV",
    "apr": "APR",
    "apy": "APY",
    "ach": "ACH",
    "atm": "ATM",
    "pfs": "PFS",
    "cd": "CD",
}


def pct(v: float) -> str:
    """6.125 -> '6.125%', 7.13 -> '7.13%', 0.1 -> '0.10%', 15.5 -> '15.50%'."""
    s = f"{v:.3f}"
    if s.endswith("0"):
        s = s[:-1]
    return s + "%"


def money(v: float) -> str:
    return f"${v:,.0f}" if abs(v - round(v)) < 1e-9 else f"${v:,.2f}"


def display_value(f: Fact) -> str:
    if f.value is None:
        return f.value_text
    if f.unit == "%":
        return pct(f.value) + (f"-{pct(f.value_hi)}" if f.value_hi is not None else "")
    if f.unit == "$":
        return money(f.value) + (f"-{money(f.value_hi)}" if f.value_hi is not None else "")
    return f"{f.value:g}"


def display_product(prod: str, family: str = "") -> str:
    name = " ".join(_ACRONYM.get(w, w.capitalize()) for w in prod.split())
    if family == "mortgage" and "mortgage" not in prod:
        name += " Mortgage"
    return name


def months_phrase(m: int) -> str:
    return f"{m // 12}-year" if m >= 120 and m % 12 == 0 else f"{m} month"


def range_phrase(r: tuple, noun: str) -> str:
    lo, hi = r
    if lo is None or lo <= 0.01:
        return f"{noun} up to {money(hi)}" if hi is not None else f"any {noun}"
    if hi is None:
        return f"{noun} of {money(lo)} or more"
    return f"{noun} {money(lo)} to {money(hi)}"


def qual_phrases(quals: dict, family: str) -> list[str]:
    out = []
    if "ira_type" in quals:
        out.append(f"{quals['ira_type'].capitalize()} IRA")
    if "purpose" in quals:
        out.append(quals["purpose"])
    if "arm" in quals:
        out.append(f"{quals['arm']} ARM")
    if "term_months" in quals:
        out.append(months_phrase(int(quals["term_months"])))
    if "model_year" in quals:
        my = quals["model_year"]
        if isinstance(my, tuple):
            out.append(f"{my[1]} and older model years" if my[0] is None else f"{my[0]}-{my[1]} model years")
        else:
            out.append(f"{my} model year")
    for name in ("balance_range", "loan_amount_range", "amount_range"):
        if isinstance(quals.get(name), tuple):
            noun = "loan amount" if name == "loan_amount_range" or family in LOAN_FAMILIES_EXCL_CARD else "balance"
            out.append(range_phrase(quals[name], noun))
    if "phase" in quals:
        out.append(
            {
                "intro": "introductory rate",
                "intro_6mo": "first 6 months promotional rate",
                "after_intro": "after the promotional period",
            }.get(str(quals["phase"]), str(quals["phase"]))
        )
    return out


def attr_name(attr: str) -> str:
    return {"apy": "APY", "apr": "APR", "rate": "rate", "dividend_rate": "dividend rate", "fee": "fee"}.get(attr, attr)
