"""Core data types shared by every stage of AASL.

Vocabulary follows Algorithm 1:

* a *text unit* is one element of U;
* a *label* is what a coder (human or model) assigns to a unit: whether it is
  in the theoretical domain D, which category of the schema C it belongs to,
  or the residual when it is in-domain but no category fits;
* a *coding* is a model label plus the audit trail needed to inspect it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Reserved label names. Category names in a schema may never use them.
RESIDUAL = "RESIDUAL"
OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
CODING_ERROR = "CODING_ERROR"
NO_SUBCATEGORY = "NONE_FITS"       # answer when no subcategory of the category fits
RESERVED_LABELS = frozenset({RESIDUAL, OUT_OF_DOMAIN, CODING_ERROR})


@dataclass(frozen=True)
class TextUnit:
    """One unit of text from U, with metadata used for stratification."""

    uid: str
    text: str
    meta: dict = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict:
        return {"uid": self.uid, "text": self.text, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict) -> "TextUnit":
        return cls(uid=str(d["uid"]), text=str(d["text"]), meta=dict(d.get("meta", {})))


@dataclass
class Label:
    """A label for one unit.

    ``category`` is ``None`` when the unit is out of domain, ``RESIDUAL`` when it
    is in-domain but unclassifiable under the current schema, and otherwise the
    name of a schema category. ``subcategory`` refines a category that defines
    subcategories (``None`` when it has none or none fits). ``attributes`` maps
    attribute name to a list of values (single-valued attributes hold at most
    one value).
    """

    in_domain: bool
    category: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    subcategory: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.in_domain:
            self.category = None
            self.attributes = {}
            self.subcategory = None
        elif self.category is None:
            raise ValueError("an in-domain label needs a category or RESIDUAL")
        elif self.category == RESIDUAL:
            self.subcategory = None
        if self.subcategory == NO_SUBCATEGORY:
            self.subcategory = None

    @property
    def flat(self) -> str:
        """Single string used for agreement and evaluation."""
        return OUT_OF_DOMAIN if not self.in_domain else str(self.category)

    def to_dict(self) -> dict:
        return {"in_domain": self.in_domain, "category": self.category,
                "subcategory": self.subcategory,
                "attributes": {k: list(v) for k, v in self.attributes.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Label":
        return cls(in_domain=bool(d["in_domain"]), category=d.get("category"),
                   attributes={k: list(v) for k, v in (d.get("attributes") or {}).items()},
                   subcategory=d.get("subcategory"))


@dataclass
class Coding:
    """A model coding of one unit, with the evidence the model cited."""

    uid: str
    label: Optional[Label]
    status: str = "ok"                  # "ok" or "error"
    error: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    evidence_verified: dict = field(default_factory=dict)
    schema_fingerprint: str = ""
    model_name: str = ""

    @property
    def flat(self) -> str:
        if self.status != "ok" or self.label is None:
            return CODING_ERROR
        return self.label.flat

    def to_dict(self) -> dict:
        return {"uid": self.uid, "label": self.label.to_dict() if self.label else None,
                "status": self.status, "error": self.error, "evidence": self.evidence,
                "evidence_verified": self.evidence_verified,
                "schema_fingerprint": self.schema_fingerprint, "model_name": self.model_name}

    @classmethod
    def from_dict(cls, d: dict) -> "Coding":
        lab = d.get("label")
        return cls(uid=str(d["uid"]), label=Label.from_dict(lab) if lab else None,
                   status=d.get("status", "ok"), error=d.get("error"),
                   evidence=d.get("evidence", {}), evidence_verified=d.get("evidence_verified", {}),
                   schema_fingerprint=d.get("schema_fingerprint", ""),
                   model_name=d.get("model_name", ""))


class AASLError(Exception):
    """Base class for pipeline errors."""


class AwaitingHumanInput(AASLError):
    """Raised when a step needs researcher input before the run can continue.

    The run state is saved; re-running after the files are completed resumes
    from the same point.
    """

    def __init__(self, message: str, paths: Optional[list] = None):
        super().__init__(message)
        self.paths = [str(p) for p in (paths or [])]


class CalibrationFailed(AASLError):
    """The inner calibration loop hit its iteration limit below the threshold."""


class ConfirmationFailed(AASLError):
    """The final schema did not reach the threshold on the held-out sample."""


class ReliabilityTooLow(AASLError):
    """Human inter-coder reliability is below the configured minimum."""


class InsufficientSupport(AASLError):
    """A sample has too few human-coded units of some category to certify it.

    Prompt revision cannot fix this, so it is raised before any model coding or
    prompt-revision request. Enlarge the sample (the extension is drawn at
    random from units not yet used) and re-run.
    """

    def __init__(self, message: str, shortfalls: list):
        super().__init__(message)
        self.shortfalls = shortfalls


class ModelUnavailable(AASLError):
    """Most model calls in a batch failed (bad key, outage, rate-limit storm)."""


def normalize_text(s: str) -> str:
    """Lowercase and collapse whitespace; used for quote checks and matching."""
    return " ".join(str(s).lower().split())


def quote_in_text(quote: Optional[str], text: str) -> bool:
    """True when ``quote`` occurs verbatim (modulo case/whitespace) in ``text``."""
    if not quote or not str(quote).strip():
        return False
    return normalize_text(quote) in normalize_text(text)
