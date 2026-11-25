"""Inductive discovery (lines 8-9 of Algorithm 1).

R <- in-domain units that M places in the residual
M proposes new categories from R, each with supporting text

Proposals are built only from residual units. Every supporting passage is
checked against the unit it claims to come from; passages that do not appear
in the unit are discarded, and a proposal left with fewer than
``min_units`` verified units is dropped before it reaches the researcher.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from .core import RESIDUAL, quote_in_text
from .llm.base import LanguageModel, call_json
from .metrics import coverage
from .prompts import propose_prompt
from .schema import Schema


@dataclass
class Evidence:
    uid: str
    quote: str
    verified: bool


@dataclass
class Proposal:
    pid: str
    name: str
    definition: str
    evidence: list = field(default_factory=list)       # verified Evidence only
    rejected_evidence: list = field(default_factory=list)
    batches: list = field(default_factory=list)

    @property
    def unit_ids(self) -> list:
        return sorted({e.uid for e in self.evidence})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unit_ids"] = self.unit_ids
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Proposal":
        return cls(pid=d["pid"], name=d["name"], definition=d["definition"],
                   evidence=[Evidence(**e) for e in d.get("evidence", [])],
                   rejected_evidence=[Evidence(**e) for e in d.get("rejected_evidence", [])],
                   batches=list(d.get("batches", [])))


@dataclass
class DiscoveryResult:
    residual_uids: list
    n_in_domain: int
    coverage: Optional[float]
    proposals: list
    dropped: list            # proposals removed by verification, with reasons

    def to_dict(self) -> dict:
        return {"residual_uids": self.residual_uids, "n_in_domain": self.n_in_domain,
                "coverage": self.coverage, "proposals": [p.to_dict() for p in self.proposals],
                "dropped": self.dropped}

    @classmethod
    def from_dict(cls, d: dict) -> "DiscoveryResult":
        return cls(residual_uids=d["residual_uids"], n_in_domain=d["n_in_domain"],
                   coverage=d["coverage"], proposals=[Proposal.from_dict(p) for p in d["proposals"]],
                   dropped=d["dropped"])


def collect_residual(codings: list) -> tuple:
    """Return (residual uids, number of in-domain units, coverage). Units with
    coding errors are excluded: they are not evidence of a gap in the schema."""
    ok = [c for c in codings if c.status == "ok" and c.label is not None]
    in_domain = [c for c in ok if c.label.in_domain]
    residual = sorted(c.uid for c in in_domain if c.label.category == RESIDUAL)
    return residual, len(in_domain), coverage([c.label.flat for c in in_domain])


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def propose_categories(model: LanguageModel, schema: Schema, codings: list, units: dict,
                       batch_size: int = 40, min_units: int = 2, retries: int = 2,
                       round_no: int = 0, rejected: Optional[list] = None) -> DiscoveryResult:
    """``rejected`` holds earlier rejections (dicts with name, definition, rationale).
    The model is told not to repeat them, and a repeat under the same name is
    dropped before it reaches the researcher."""
    residual, n_dom, cov = collect_residual(codings)
    proposals: dict = {}
    dropped = []
    existing = {_key(n) for n in schema.category_names}
    turned_down = {_key(r["name"]) for r in (rejected or [])}
    for b, start in enumerate(range(0, len(residual), batch_size)):
        batch_ids = residual[start:start + batch_size]
        if len(batch_ids) < min_units:
            continue
        batch = [units[u] for u in batch_ids]
        p = propose_prompt(schema, batch, rejected)
        try:
            reply = call_json(model, p.system, p.user, p.schema, p.task, retries=retries)
        except Exception as exc:
            dropped.append({"name": None, "batch": b, "reason": f"proposal call failed: {exc}"})
            continue
        allowed = set(batch_ids)
        for item in reply["proposals"]:
            name = " ".join(str(item["name"]).split())
            k = _key(name)
            if not k:
                continue
            good, bad = [], []
            for s in item["supporting"]:
                uid = str(s["unit_id"])
                ok = uid in allowed and quote_in_text(s["quote"], units[uid].text)
                (good if ok else bad).append(Evidence(uid, str(s["quote"]), ok))
            if k in existing:
                dropped.append({"name": name, "batch": b,
                                "reason": "duplicates an existing category"})
                continue
            if k in turned_down:
                dropped.append({"name": name, "batch": b,
                                "reason": "already rejected by the researcher in an earlier round"})
                continue
            prop = proposals.get(k)
            if prop is None:
                prop = Proposal(pid=f"r{round_no}-p{len(proposals) + 1}", name=name,
                                definition=str(item["definition"]).strip())
                proposals[k] = prop
            seen = {e.uid for e in prop.evidence}
            prop.evidence += [e for e in good if e.uid not in seen]
            prop.rejected_evidence += bad
            prop.batches.append(b)
    kept = []
    for prop in proposals.values():
        if len(prop.unit_ids) < min_units:
            dropped.append({"name": prop.name, "batch": prop.batches,
                            "reason": f"{len(prop.unit_ids)} verified unit(s) < {min_units}; "
                                      f"{len(prop.rejected_evidence)} passage(s) not found in "
                                      "the cited units"})
        else:
            kept.append(prop)
    return DiscoveryResult(residual, n_dom, cov, kept, dropped)
