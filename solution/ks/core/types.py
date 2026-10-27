"""Result types returned by every answering layer, and the institution's voice used in their wording."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceRef:
    url: str
    snapshot_id: str | None = None
    quote: str | None = None


@dataclass
class Result:
    """What the answerers return; `solution.py` converts it into the contract's `Answer`."""
    text: str
    sources: list = field(default_factory=list)
    confident: bool = True
    conflict_note: str | None = None
    extractive: bool = False
    quotes: list = field(default_factory=list)


@dataclass(frozen=True)
class Voice:
    """How answers refer to the institution: by name when the data names it, neutral wording otherwise."""
    name: str | None
    contact_channels: str = ""

    def label(self, default: str = "the institution") -> str:
        return self.name or default

    def poss(self) -> str:
        return f"{self.name}'s" if self.name else "The institution's"

    def contact_phrase(self) -> str:
        if self.name:
            return f"Please contact {self.name} directly ({self.contact_channels}) to confirm."
        return "Please contact the institution directly to confirm."

    def refusal_text(self) -> str:
        where = (
            f"the information {self.name} publishes on its website"
            if self.name
            else "the information published on the website"
        )
        return f"I don't know. I couldn't find that in {where}. {self.contact_phrase()}"

    def refusal(self, note: str | None = None) -> Result:
        """A no-number refusal (the scorer requires zero digits beyond those in the question)."""
        return Result(self.refusal_text(), [], False, note)
