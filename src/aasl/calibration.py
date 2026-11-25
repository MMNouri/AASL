"""Deductive calibration (lines 4-7 of Algorithm 1).

repeat
    M codes A into a category in C or into a residual
    if precision or recall < tau then revise the prompts
until precision and recall >= tau

Revising the prompts means changing how C is *explained* to the model (notes,
wording of definitions). It cannot add or remove categories; that happens only
through discovery and researcher adjudication, so the two kinds of change stay
separate in the record.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from .core import AwaitingHumanInput, CalibrationFailed
from .metrics import EvaluationReport, disagreements, evaluate
from .schema import Schema


class PromptReviser(ABC):
    @abstractmethod
    def revise(self, schema: Schema, report: EvaluationReport, errors: list, tag: str) -> Schema:
        """Return a schema with revised prompt notes/definitions (same categories)."""


@dataclass
class CalibrationResult:
    schema: Schema
    report: EvaluationReport
    codings: list
    iterations: list = field(default_factory=list)   # one summary per pass


def calibrate(schema: Schema, sample: list, gold: dict, code: Callable, reviser: PromptReviser,
              tau: float, min_support: int, max_iterations: int, tag: str,
              gate_attributes: bool = False, gate_subcategories: bool = False,
              cluster_of: Optional[dict] = None, on_iteration: Optional[Callable] = None) -> CalibrationResult:
    """Run the inner loop. ``code(schema, units, name)`` returns codings."""
    units = {u.uid: u for u in sample}
    iterations = []
    for it in range(1, max_iterations + 1):
        codings = code(schema, sample, f"{tag}_iter{it}")                     # line 5
        report = evaluate(gold, codings, schema, tau, min_support,
                          gate_attributes=gate_attributes, gate_subcategories=gate_subcategories,
                          cluster_of=cluster_of)
        iterations.append({"iteration": it, "schema_version": schema.version,
                           "fingerprint": schema.fingerprint(), "passes": report.passes,
                           "failures": list(report.failures)})
        if on_iteration:
            on_iteration(it, schema, report)
        if report.passes:                                                     # line 7
            return CalibrationResult(schema, report, codings, iterations)
        if it == max_iterations:
            break
        errors = disagreements(gold, codings, units)
        revised = reviser.revise(schema, report, errors, f"{tag}_iter{it}")   # line 6
        if revised.category_names != schema.category_names:
            raise ValueError("prompt revision changed the category set; add categories "
                             "only through discovery and adjudication")
        if revised.fingerprint() == schema.fingerprint():
            raise ValueError("prompt revision returned an unchanged schema; the loop would "
                             "repeat the same failing pass")
        schema = revised
    raise CalibrationFailed(f"{tag}: below tau={tau} after {max_iterations} passes: "
                            + "; ".join(report.failures[:5]))


# --------------------------------------------------------------------- revisers
class CallbackReviser(PromptReviser):
    def __init__(self, fn: Callable):
        self.fn = fn

    def revise(self, schema, report, errors, tag):
        return self.fn(schema, report, errors, tag)


class FileReviser(PromptReviser):
    """Writes the failing report and the disagreements, then waits for the
    researcher to edit ``revision.yaml`` and set ``ready: true``."""

    def __init__(self, root):
        self.root = Path(root)

    def revise(self, schema, report, errors, tag):
        folder = self.root / tag
        path = folder / "revision.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if data.get("ready") is True:
                notes = {k: [str(x) for x in (v or [])] for k, v in (data.get("notes") or {}).items()}
                definitions = {k: str(v) for k, v in (data.get("definitions") or {}).items()}
                unknown = set(definitions) - set(schema.category_names)
                if unknown:
                    raise ValueError(f"{path}: definitions for unknown categories {sorted(unknown)}")
                return schema.with_notes(notes, definitions,
                                         reason=str(data.get("reason") or f"calibration {tag}"))
            raise AwaitingHumanInput(f"Prompt revision pending for '{tag}': edit {path} and set "
                                     "ready: true.", [path])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.md").write_text(report.to_markdown(f"Calibration {tag}"), encoding="utf-8")
        with open(folder / "disagreements.csv", "w", newline="", encoding="utf-8") as fh:
            cols = ["uid", "human", "model", "model_evidence", "error", "text"]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in errors:
                w.writerow({k: row.get(k) for k in cols})
        template = {
            "ready": False,
            "reason": "",
            "notes": {k: list(v) for k, v in schema.notes.items()} or
                     {"domain": [], "categories": [], **{c: [] for c in schema.category_names}},
            "definitions": {c.name: c.definition for c in schema.categories},
        }
        header = ("# Revise how the schema is explained to the model, guided by report.md and\n"
                  "# disagreements.csv. Notes: keys 'domain', 'categories', a category name, or\n"
                  "# 'attr:<attribute>'. You may reword definitions but not add or remove\n"
                  "# categories. Set ready: true when done, then re-run.\n")
        path.write_text(header + yaml.safe_dump(template, sort_keys=False, allow_unicode=True,
                                                width=100), encoding="utf-8")
        raise AwaitingHumanInput(f"Calibration '{tag}' is below tau. Review "
                                 f"{folder / 'report.md'} and edit {path}.", [path])
