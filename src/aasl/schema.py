"""The category schema C (the codebook).

A schema holds the theoretical domain D, the categories, the attributes that
are extracted from in-domain units, and the *prompt notes* that calibration
adds when the model misreads a definition. Notes are kept separate from the
theory-derived definitions so a reader can see what theory supplied and what
calibration added.

Every change produces a new version; ``fingerprint()`` identifies the exact
content a model coding was produced under.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .core import NO_SUBCATEGORY, RESERVED_LABELS

ATTRIBUTE_KINDS = ("single", "multi", "text")


@dataclass
class Subcategory:
    name: str
    definition: str


@dataclass
class Category:
    name: str
    definition: str
    include: list = field(default_factory=list)
    exclude: list = field(default_factory=list)
    examples: list = field(default_factory=list)
    origin: str = "theory"          # "theory" (seed) or "discovered"
    source: str = ""                # citation for seeds, provenance for discovered
    added_in_round: int = 0
    subcategories: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.subcategories = [s if isinstance(s, Subcategory) else Subcategory(**s)
                              for s in self.subcategories]
        names = [s.name for s in self.subcategories]
        if len(names) != len(set(names)):
            raise ValueError(f"category {self.name!r}: subcategory names must be unique")
        for n in names:
            if not n.strip() or n in RESERVED_LABELS or n == NO_SUBCATEGORY:
                raise ValueError(f"category {self.name!r}: invalid subcategory name {n!r}")

    @property
    def subcategory_names(self) -> list:
        return [s.name for s in self.subcategories]


@dataclass
class Attribute:
    name: str
    description: str
    kind: str = "single"            # single | multi | text
    options: list = field(default_factory=list)
    applies_to: list = field(default_factory=list)   # empty = all categories

    def __post_init__(self) -> None:
        if self.kind not in ATTRIBUTE_KINDS:
            raise ValueError(f"attribute {self.name!r}: kind must be one of {ATTRIBUTE_KINDS}")
        if self.kind in ("single", "multi") and not self.options:
            raise ValueError(f"attribute {self.name!r}: kind {self.kind!r} needs options")

    def applies(self, category: str) -> bool:
        return not self.applies_to or category in self.applies_to


@dataclass
class Domain:
    name: str
    definition: str
    include: list = field(default_factory=list)
    exclude: list = field(default_factory=list)


@dataclass
class Schema:
    domain: Domain
    categories: list = field(default_factory=list)
    attributes: list = field(default_factory=list)
    # Calibration notes: key "domain", "categories" or a category name.
    notes: dict = field(default_factory=dict)
    version: int = 1
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ queries
    @property
    def category_names(self) -> list:
        return [c.name for c in self.categories]

    def category(self, name: str) -> Category:
        for c in self.categories:
            if c.name == name:
                return c
        raise KeyError(name)

    def attributes_for(self, category: str) -> list:
        return [a for a in self.attributes if a.applies(category)]

    def notes_for(self, key: str) -> list:
        return list(self.notes.get(key, []))

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        names = [c.name for c in self.categories]
        if len(names) != len(set(names)):
            raise ValueError("category names must be unique")
        for n in names:
            if not n or not n.strip():
                raise ValueError("empty category name")
            if n in RESERVED_LABELS:
                raise ValueError(f"{n!r} is reserved and cannot be a category name")
        anames = [a.name for a in self.attributes]
        if len(anames) != len(set(anames)):
            raise ValueError("attribute names must be unique")
        for a in self.attributes:
            unknown = set(a.applies_to) - set(names)
            if unknown:
                raise ValueError(f"attribute {a.name!r} applies to unknown categories {sorted(unknown)}")

    # ------------------------------------------------------------------ changes
    def _bump(self, event: dict) -> "Schema":
        new = copy.deepcopy(self)
        new.version = self.version + 1
        new.history = list(self.history) + [dict(event, version=new.version)]
        new.validate()
        return new

    def with_notes(self, notes: dict, definitions: Optional[dict] = None, reason: str = "") -> "Schema":
        """Return a new version with revised prompt notes (and optionally
        reworded definitions). The category set itself cannot change here:
        adding categories is the job of discovery and adjudication."""
        new = self._bump({"event": "prompt_revision", "reason": reason})
        new.notes = {k: list(v) for k, v in notes.items()}
        for name, text in (definitions or {}).items():
            new.category(name).definition = text
        return new

    def with_category(self, cat: Category, reason: str = "") -> "Schema":
        if cat.name in self.category_names:
            raise ValueError(f"category {cat.name!r} already exists")
        new = self._bump({"event": "category_added", "category": cat.name, "reason": reason})
        new.categories.append(copy.deepcopy(cat))
        new.validate()
        return new

    def with_merge(self, target: str, include: list, examples: list, reason: str = "") -> "Schema":
        """Widen an existing category with material from a merged proposal."""
        new = self._bump({"event": "category_widened", "category": target, "reason": reason})
        cat = new.category(target)
        cat.include = list(dict.fromkeys(list(cat.include) + list(include)))
        cat.examples = list(dict.fromkeys(list(cat.examples) + list(examples)))
        return new

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "domain": asdict(self.domain),
            "categories": [asdict(c) for c in self.categories],
            "attributes": [asdict(a) for a in self.attributes],
            "notes": {k: list(v) for k, v in self.notes.items()},
            "version": self.version,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(
            domain=Domain(**d["domain"]),
            categories=[Category(**c) for c in d.get("categories", [])],
            attributes=[Attribute(**a) for a in d.get("attributes", [])],
            notes={k: list(v) for k, v in (d.get("notes") or {}).items()},
            version=int(d.get("version", 1)),
            history=list(d.get("history", [])),
        )

    def fingerprint(self) -> str:
        """Hash of everything that reaches a prompt (not the history)."""
        content = self.to_dict()
        content.pop("history")
        content.pop("version")
        blob = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=100)
        path.write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Schema":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        """Human-readable codebook for coders."""
        out = [f"# Codebook: {self.domain.name} (schema v{self.version})", "",
               "## Domain", self.domain.definition, ""]
        if self.domain.include:
            out += ["In the domain:"] + [f"- {x}" for x in self.domain.include] + [""]
        if self.domain.exclude:
            out += ["Not in the domain:"] + [f"- {x}" for x in self.domain.exclude] + [""]
        out += ["## Categories", ""]
        for c in self.categories:
            out += [f"### {c.name}", c.definition, ""]
            if c.subcategories:
                out += ["Subcategories (give one, or leave blank if none fits):"]
                out += [f"- {sc.name}: {sc.definition}" for sc in c.subcategories] + [""]
            if c.include:
                out += ["Includes:"] + [f"- {x}" for x in c.include] + [""]
            if c.exclude:
                out += ["Excludes:"] + [f"- {x}" for x in c.exclude] + [""]
            for n in self.notes_for(c.name):
                out.append(f"- Note: {n}")
            out.append("")
        out += [f"### {RESERVED_RESIDUAL_HELP}", ""]
        if self.attributes:
            out += ["## Attributes", ""]
            for a in self.attributes:
                opts = f" Options: {', '.join(a.options)}." if a.options else ""
                scope = f" Applies to: {', '.join(a.applies_to)}." if a.applies_to else ""
                out.append(f"- **{a.name}** ({a.kind}): {a.description}.{opts}{scope}")
        return "\n".join(out) + "\n"


RESERVED_RESIDUAL_HELP = ("RESIDUAL: use when the unit is in the domain but none of the "
                          "categories above fits. Do not force a fit.")
