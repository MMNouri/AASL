"""Researcher adjudication (lines 10-11 of Algorithm 1).

Researcher keeps, merges, or rejects each proposal
C <- C U {kept categories}

Every proposal needs a decision and a written rationale; the decisions are
kept in a log that documents how the schema grew. Actions:

* ``keep``   – add the proposal as a new category (optionally renamed or
  with a reworded definition).
* ``merge``  – fold the proposal into an existing category (widening it) or
  into another proposal that is kept in the same round.
* ``reject`` – leave it out; its units stay in the residual.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

from .core import AwaitingHumanInput
from .schema import Category, Schema

ACTIONS = ("keep", "merge", "reject")


@dataclass
class Decision:
    pid: str
    action: str
    rationale: str
    into: Optional[str] = None          # merge target: category name or proposal pid
    name: Optional[str] = None          # keep: final name (defaults to proposal name)
    definition: Optional[str] = None    # keep: final definition
    source: Optional[str] = None        # keep: theory reference supporting the category

    def to_dict(self) -> dict:
        return asdict(self)


class Adjudicator(ABC):
    @abstractmethod
    def decide(self, proposals: list, schema: Schema, tag: str) -> list:
        """Return one Decision per proposal."""


def check_decisions(proposals: list, decisions: list, schema: Schema) -> list:
    problems = []
    pids = {p.pid for p in proposals}
    by_pid = {d.pid: d for d in decisions}
    for p in proposals:
        if p.pid not in by_pid:
            problems.append(f"{p.pid} ({p.name}): no decision")
    for d in decisions:
        if d.pid not in pids:
            problems.append(f"{d.pid}: no such proposal")
        if d.action not in ACTIONS:
            problems.append(f"{d.pid}: action must be one of {ACTIONS}, got {d.action!r}")
        if not (d.rationale or "").strip():
            problems.append(f"{d.pid}: a rationale is required")
        if d.action == "merge":
            target = by_pid.get(d.into)
            if d.into in schema.category_names:
                pass
            elif target is not None and target.action == "keep":
                pass
            else:
                problems.append(f"{d.pid}: merge target {d.into!r} must be an existing category "
                                "or a proposal kept in this round")
    return problems


def apply_decisions(schema: Schema, proposals: list, decisions: list, round_no: int) -> tuple:
    """Return (new schema, names of kept categories, decision log entries)."""
    problems = check_decisions(proposals, decisions, schema)
    if problems:
        raise ValueError("invalid decisions: " + "; ".join(problems))
    by_pid = {p.pid: p for p in proposals}
    kept_names, log = [], []
    final_name = {}

    for d in decisions:                                   # keeps first
        if d.action != "keep":
            continue
        p = by_pid[d.pid]
        merged_in = [by_pid[x.pid] for x in decisions if x.action == "merge" and x.into == d.pid]
        examples = [e.quote for e in p.evidence[:3]]
        include = [m.definition for m in merged_in]
        for m in merged_in:
            examples += [e.quote for e in m.evidence[:2]]
        cat = Category(name=(d.name or p.name).strip(),
                       definition=(d.definition or p.definition).strip(),
                       include=include, examples=examples, origin="discovered",
                       source=d.source or f"discovered in round {round_no} ({p.pid})",
                       added_in_round=round_no)
        schema = schema.with_category(cat, reason=d.rationale)
        kept_names.append(cat.name)
        final_name[d.pid] = cat.name

    for d in decisions:                                   # merges into existing categories
        if d.action == "merge" and d.into in schema.category_names and d.into not in final_name:
            p = by_pid[d.pid]
            schema = schema.with_merge(d.into, include=[p.definition],
                                       examples=[e.quote for e in p.evidence[:2]],
                                       reason=d.rationale)

    for d in decisions:
        p = by_pid[d.pid]
        target = final_name.get(d.into, d.into) if d.action == "merge" else None
        log.append({"round": round_no, "pid": d.pid, "proposal": p.name,
                    "definition": p.definition, "n_units": len(p.unit_ids),
                    "action": d.action, "into": target,
                    "final_name": final_name.get(d.pid), "rationale": d.rationale})
    return schema, kept_names, log


# ------------------------------------------------------------- adjudicators
class CallbackAdjudicator(Adjudicator):
    def __init__(self, fn: Callable):
        self.fn = fn

    def decide(self, proposals, schema, tag):
        return self.fn(proposals, schema, tag)


class FileAdjudicator(Adjudicator):
    """Writes ``proposals.yaml``; the researcher fills in the decision fields."""

    def __init__(self, root):
        self.root = Path(root)

    def decide(self, proposals, schema, tag):
        if not proposals:
            return []
        path = self.root / tag / "proposals.yaml"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            doc = {"ready": False, "existing_categories": schema.category_names, "proposals": []}
            for p in proposals:
                doc["proposals"].append({
                    "pid": p.pid, "name": p.name, "definition": p.definition,
                    "n_units": len(p.unit_ids),
                    "supporting_text": [f"[{e.uid}] {e.quote}" for e in p.evidence[:8]],
                    "decision": {"action": "", "rationale": "", "into": "", "name": "",
                                 "definition": "", "source": ""},
                })
            header = ("# For each proposal set decision.action to keep, merge or reject and give a\n"
                      "# rationale grounded in theory. For merge, set 'into' to an existing\n"
                      "# category or to the pid of a proposal you keep. For keep, you may set\n"
                      "# a final name/definition and the theory source. Then set ready: true.\n")
            path.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                                                    width=100), encoding="utf-8")
            raise AwaitingHumanInput(f"{len(proposals)} candidate categories need adjudication: "
                                     f"edit {path}.", [path])
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if doc.get("ready") is not True:
            raise AwaitingHumanInput(f"Adjudication pending: complete {path} and set ready: true.",
                                     [path])
        decisions = []
        for item in doc.get("proposals", []):
            d = item.get("decision") or {}
            decisions.append(Decision(pid=item["pid"], action=str(d.get("action") or "").strip(),
                                      rationale=str(d.get("rationale") or "").strip(),
                                      into=(str(d["into"]).strip() or None) if d.get("into") else None,
                                      name=(str(d["name"]).strip() or None) if d.get("name") else None,
                                      definition=(str(d["definition"]).strip() or None)
                                      if d.get("definition") else None,
                                      source=(str(d["source"]).strip() or None)
                                      if d.get("source") else None))
        problems = check_decisions(proposals, decisions, schema)
        if problems:
            raise AwaitingHumanInput(f"{path} has problems: " + "; ".join(problems), [path])
        return decisions
