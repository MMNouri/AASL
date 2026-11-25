"""Algorithm 1 (AI-Augmented Supervised Learning) end to end.

Input : text units U; theoretical domain D; seed categories S from theory;
        language model M; threshold tau
Output: final category schema C*; coded text units X

 1: C <- S
 2: repeat
 3:     code a sample A by hand using C
 4:     repeat
 5:         M codes A into a category in C or into a residual
 6:         if precision or recall < tau then revise the prompts
 7:     until precision and recall >= tau
 8:     R <- in-domain units that M places in the residual
 9:     M proposes new categories from R, each with supporting text
10:     researcher keeps, merges, or rejects each proposal
11:     C <- C U {kept categories}
12: until no new category is kept
13: confirm precision and recall >= tau on a held-out sample
14: C* <- C;  X <- M codes all of U using C*
15: return C*, X

The comments ``# line N`` below mark where each line is carried out.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from .adjudication import Adjudicator, apply_decisions
from .annotation import AnnotationResult, Annotator
from .calibration import PromptReviser, calibrate
from .coder import Coder
from .core import (AASLError, Coding, ConfirmationFailed, InsufficientSupport, Label,
                   ReliabilityTooLow)
from .discovery import DiscoveryResult, propose_categories
from .llm.base import LanguageModel
from .metrics import evaluate, support_shortfalls
from .sampling import SampleLedger
from .schema import Schema
from .store import RunStore


@dataclass
class Settings:
    tau: float = 0.9
    min_support: int = 10
    gate_attributes: bool = False
    gate_subcategories: bool = False
    # human reliability
    min_reliability: float = 0.7
    reliability_metric: str = "alpha_nominal"      # or cohen_kappa (two coders)
    on_low_reliability: str = "stop"               # stop | warn
    # samples
    holdout_size: int = 300
    calibration_size: int = 300
    discovery_pool_size: int = 0                   # extra units coded for line 8
    strata: list = field(default_factory=list)
    cluster: Optional[str] = None
    max_per_cluster: Optional[int] = None
    seed: int = 0
    # loops
    max_rounds: int = 5
    max_calibration_iterations: int = 6
    # discovery
    proposal_batch_size: int = 40
    min_proposal_units: int = 2
    # model calls
    max_workers: int = 8
    retries: int = 2
    backoff: float = 1.0
    # held-out confirmation
    n_boot: int = 1000

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        unknown = set(d or {}) - set(known)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        return cls(**known)


@dataclass
class RunResult:
    schema: Schema                  # C*
    codings: list                   # X
    holdout_report: object
    rounds: list


class AASLRun:
    def __init__(self, units: list, seed_schema: Schema, model: LanguageModel,
                 annotator: Annotator, reviser: PromptReviser, adjudicator: Adjudicator,
                 settings: Settings, run_dir, verbose: bool = True):
        ids = [u.uid for u in units]
        if len(ids) != len(set(ids)):
            raise ValueError("unit ids must be unique")
        self.units = list(units)
        self.by_uid = {u.uid: u for u in self.units}
        self.seed_schema = seed_schema
        self.model = model
        self.annotator = annotator
        self.reviser = reviser
        self.adjudicator = adjudicator
        self.s = settings
        self.store = RunStore(run_dir)
        self.ledger = SampleLedger(self.store.path("samples.json"))
        self.verbose = verbose
        self.cluster_of = ({u.uid: u.meta.get(settings.cluster, u.uid) for u in self.units}
                           if settings.cluster else None)

    # --------------------------------------------------------------- helpers
    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _draw_kwargs(self) -> dict:
        return dict(strata=self.s.strata, cluster=self.s.cluster,
                    max_per_cluster=self.s.max_per_cluster,
                    seed=self.s.seed + len(self.ledger.samples))

    def _draw(self, name: str, n: int) -> list:
        """Draw (or extend) a sample that needs no human coding; returns its units."""
        parts = self.ledger.draw_extended(name, self.units, n, **self._draw_kwargs())
        return [u for _, units in parts for u in units]

    def _code(self, schema: Schema, units: list, name: str) -> list:
        coder = Coder(self.model, schema, retries=self.s.retries, max_workers=self.s.max_workers,
                      backoff=self.s.backoff)
        path = self.store.path("codings", f"{name}__{schema.fingerprint()}.jsonl")
        return coder.code_many(units, checkpoint=path, progress=self.verbose)

    def _save_schema(self, schema: Schema) -> None:
        schema.save(self.store.path("schemas", f"schema_v{schema.version}.yaml"))
        self.store.save("schema_current", schema.to_dict())

    def _annotate(self, units: list, schema: Schema, tag: str) -> AnnotationResult:
        key = f"gold/{tag}"
        if self.store.has(key):
            d = self.store.load(key)
            return AnnotationResult({u: Label.from_dict(v) for u, v in d["gold"].items()},
                                    d["reliability"], d["n_disagreements"])
        res = self.annotator.annotate(units, schema, tag)
        value = res.reliability.get(self.s.reliability_metric)
        if value is None:
            self.store.log("reliability_unavailable", tag=tag, reliability=res.reliability)
        elif value < self.s.min_reliability:
            msg = (f"{tag}: {self.s.reliability_metric} = {value:.3f} < {self.s.min_reliability}. "
                   "Tighten the codebook and have the coders re-code before calibrating.")
            if self.s.on_low_reliability == "stop":
                self.store.log("reliability_too_low", tag=tag, reliability=res.reliability)
                raise ReliabilityTooLow(msg)
            self.store.log("reliability_warning", tag=tag, message=msg)
        self.store.save(key, {"gold": {u: lab.to_dict() for u, lab in res.gold.items()},
                              "reliability": res.reliability,
                              "n_disagreements": res.n_disagreements})
        self.store.log("human_coding_done", tag=tag, reliability=res.reliability,
                       n_disagreements=res.n_disagreements)
        return res

    def _annotated_sample(self, name: str, n: int, schema: Schema, tag: str) -> tuple:
        """Draw sample ``name`` (extending it if ``n`` grew), have humans code any part
        not yet coded, and return (units, merged AnnotationResult). Extensions are
        coded and reliability-checked as their own parts."""
        parts = self.ledger.draw_extended(name, self.units, n, **self._draw_kwargs())
        units, gold, rels, disagreements = [], {}, [], 0
        for k, (_part, part_units) in enumerate(parts):
            res = self._annotate(part_units, schema, tag if k == 0 else f"{tag}_ext{k}")
            units += part_units
            gold.update(res.gold)
            rels.append(res.reliability)
            disagreements += res.n_disagreements
        metric = self.s.reliability_metric
        values = [r[metric] for r in rels if r.get(metric) is not None]
        reliability = dict(rels[0]) if len(rels) == 1 else {
            metric: min(values) if values else None, "parts": rels}
        return units, AnnotationResult(gold, reliability, disagreements)

    def _require_support(self, gold: dict, schema: Schema, sample_key: str, tag: str) -> None:
        """Stop, before any model coding, when the human labels cannot certify a
        category. Prompt revision cannot create units; the sample must grow."""
        short = support_shortfalls(gold, schema, self.s.min_support)
        if not short:
            return
        lines = []
        for x in short:
            need = (f"about {x['needed']} units needed at the observed rate" if x["needed"]
                    else "not seen at all, so a much larger sample is needed")
            lines.append(f"  - {x['label']}: {x['support']} of {x['n_units']} human-coded units "
                         f"(< {self.s.min_support}); {need}")
        biggest = max([x["needed"] for x in short if x["needed"]] or [0])
        hint = (f" Set {sample_key} to at least {biggest} in the config and run again: the units "
                "already coded are kept and only the added units need coding."
                if biggest else f" Raise {sample_key} substantially, or lower min_support.")
        self.store.log("insufficient_support", tag=tag, shortfalls=short)
        raise InsufficientSupport(
            f"{tag}: the human-coded sample cannot certify every category at min_support="
            f"{self.s.min_support}:\n" + "\n".join(lines) + "\n" + hint, short)

    def _rejected(self) -> list:
        """Proposals the researcher rejected in earlier rounds (deduplicated)."""
        path = self.store.root / "decisions.jsonl"
        out, seen = [], set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                if e["action"] == "reject" and (e["round"], e["pid"]) not in seen:
                    seen.add((e["round"], e["pid"]))
                    out.append({"name": e["proposal"], "definition": e["definition"],
                                "rationale": e["rationale"]})
        return out

    def _append_decisions(self, log: list) -> None:
        path = self.store.path("decisions.jsonl")
        have = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                have.add((e["round"], e["pid"]))
        with open(path, "a", encoding="utf-8") as fh:
            for entry in log:
                if (entry["round"], entry["pid"]) not in have:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _guard_inputs(self) -> None:
        """A run is tied to one seed schema and one corpus; changing either midway
        would silently mix incompatible results."""
        corpus = hashlib.sha256()
        for u in sorted(self.units, key=lambda x: x.uid):
            corpus.update(u.uid.encode("utf-8") + b"\0" + u.text.encode("utf-8") + b"\0")
        current = {"seed_schema": self.seed_schema.fingerprint(), "corpus": corpus.hexdigest()[:16]}
        if self.store.has("run_inputs"):
            saved = self.store.load("run_inputs")
            changed = [k.replace("_", " ") for k in current if saved.get(k) != current[k]]
            if changed:
                raise AASLError(
                    f"The {' and '.join(changed)} changed since this run started "
                    f"({self.store.root}). Start a new run directory, or restore the original.")
        else:
            self.store.save("run_inputs", current)

    # ------------------------------------------------------------------- run
    def run(self) -> RunResult:
        s = self.s
        self._guard_inputs()
        # Reserve the held-out sample before anything else, so no unit used to
        # confirm the schema can influence how the schema is built.
        self._draw(SampleLedger.HOLDOUT, s.holdout_size)

        schema = self.seed_schema                                             # line 1
        if not self.store.has("schemas_seed"):
            self.store.save("schemas_seed", schema.to_dict())
            self._save_schema(schema)
        rounds = []
        r = 0
        while True:                                                           # line 2
            r += 1
            tag = f"round{r}"
            if self.store.has(f"{tag}/result"):            # round finished earlier
                done = self.store.load(f"{tag}/result")
                schema = Schema.from_dict(done["schema"])
                rounds.append(done["summary"])
                if not done["summary"]["kept"]:
                    break
                continue
            if r > s.max_rounds:
                raise AASLError(f"no stable schema after {s.max_rounds} rounds; raise max_rounds "
                                "or tighten adjudication")
            self._say(f"[round {r}] schema v{schema.version}: {schema.category_names}")

            sample, ann = self._annotated_sample(f"{tag}/calibration", s.calibration_size,
                                                 schema, f"{tag}_calibration")     # line 3
            self._require_support(ann.gold, schema, "calibration_size", f"{tag}_calibration")

            cal = calibrate(schema, sample, ann.gold, self._code, self.reviser,  # lines 4-7
                            tau=s.tau, min_support=s.min_support,
                            max_iterations=s.max_calibration_iterations, tag=f"{tag}_calibration",
                            gate_attributes=s.gate_attributes,
                            gate_subcategories=s.gate_subcategories, cluster_of=self.cluster_of,
                            on_iteration=lambda i, sc, rep: self._say(
                                f"  calibration pass {i}: {'PASS' if rep.passes else 'FAIL'}"))
            schema = cal.schema
            self._save_schema(schema)
            self.store.path("reports", f"{tag}_calibration.md").write_text(
                cal.report.to_markdown(f"Round {r} calibration"), encoding="utf-8")

            pool = list(cal.codings)
            if s.discovery_pool_size:
                extra = self._draw(f"{tag}/discovery", s.discovery_pool_size)
                pool += self._code(schema, extra, f"{tag}_discovery")
            if self.store.has(f"{tag}/discovery"):
                disc = DiscoveryResult.from_dict(self.store.load(f"{tag}/discovery"))
            else:
                disc = propose_categories(self.model, schema, pool, self.by_uid,      # lines 8-9
                                          batch_size=s.proposal_batch_size,
                                          min_units=s.min_proposal_units, retries=s.retries,
                                          round_no=r, rejected=self._rejected())
                self.store.save(f"{tag}/discovery", disc.to_dict())
            self._say(f"  residual: {len(disc.residual_uids)} of {disc.n_in_domain} in-domain "
                      f"units; proposals: {[p.name for p in disc.proposals]}")

            decisions = self.adjudicator.decide(disc.proposals, schema,              # line 10
                                                f"{tag}_adjudication")
            schema, kept, log = apply_decisions(schema, disc.proposals, decisions, r)  # line 11
            self._save_schema(schema)
            self._append_decisions(log)

            summary = {"round": r, "calibration_passes": len(cal.iterations),
                       "calibration": cal.iterations, "reliability": ann.reliability,
                       "n_residual": len(disc.residual_uids), "n_in_domain": disc.n_in_domain,
                       "coverage": disc.coverage, "n_proposals": len(disc.proposals),
                       "dropped_proposals": disc.dropped, "kept": kept,
                       "schema_version": schema.version}
            self.store.save(f"{tag}/result", {"schema": schema.to_dict(), "summary": summary})
            self.store.log("round_done", **summary)
            rounds.append(summary)
            self._say(f"  kept: {kept or 'none'}")
            if not kept:                                                      # line 12
                break

        final = schema                                                       # C*
        holdout, h_ann = self._annotated_sample(SampleLedger.HOLDOUT, s.holdout_size, final,
                                                "holdout")                       # line 13
        # Support is a property of the human labels alone, so it is checked before the
        # model codes the held-out units: extending the sample never depends on how
        # the model performed.
        self._require_support(h_ann.gold, final, "holdout_size", "holdout")
        h_codes = self._code(final, holdout, "holdout")
        report = evaluate(h_ann.gold, h_codes, final, s.tau, s.min_support,
                          gate_attributes=s.gate_attributes,
                          gate_subcategories=s.gate_subcategories,
                          cluster_of=self.cluster_of, n_boot=s.n_boot, seed=s.seed)
        self.store.path("reports", "holdout_confirmation.md").write_text(
            report.to_markdown("Held-out confirmation"), encoding="utf-8")
        self.store.save("holdout_report", report.to_dict())
        if not report.passes:
            self.store.log("confirmation_failed", failures=report.failures)
            raise ConfirmationFailed("held-out confirmation failed; see "
                                     "reports/holdout_confirmation.md: " +
                                     "; ".join(report.failures[:5]))
        self._say("held-out confirmation: PASS")

        codings = self._code(final, self.units, "final")                      # line 14
        self._write_outputs(final, codings, rounds)
        self.store.log("run_complete", schema_version=final.version, n_units=len(codings))
        return RunResult(final, codings, report, rounds)                     # line 15

    # --------------------------------------------------------------- outputs
    def _write_outputs(self, schema: Schema, codings: list, rounds: list) -> None:
        schema.save(self.store.path("output", "final_schema.yaml"))
        self.store.path("output", "codebook.md").write_text(schema.to_markdown(), encoding="utf-8")
        with open(self.store.path("output", "coded_units.jsonl"), "w", encoding="utf-8") as fh:
            for c in codings:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        attrs = [a.name for a in schema.attributes]
        with open(self.store.path("output", "coded_units.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["uid", "status", "in_domain", "category", "subcategory"] + attrs +
                       ["evidence", "evidence_verified", "schema_version", "model", "error"])
            for c in codings:
                lab = c.label
                w.writerow([c.uid, c.status,
                            "" if lab is None else ("yes" if lab.in_domain else "no"),
                            "" if lab is None or lab.category is None else lab.category,
                            "" if lab is None or lab.subcategory is None else lab.subcategory]
                           + [";".join(lab.attributes.get(a, [])) if lab else "" for a in attrs]
                           + [c.evidence.get("assign") or c.evidence.get("detect") or "",
                              all(c.evidence_verified.values()) if c.evidence_verified else "",
                              schema.version, c.model_name, c.error or ""])
        self.store.path("output", "rounds.json").write_text(json.dumps(rounds, indent=1),
                                                            encoding="utf-8")
        self.store.path("output", "audit.json").write_text(
            json.dumps(audit(schema, codings, len(rounds)), indent=1), encoding="utf-8")


def audit(schema: Schema, codings: list, n_rounds: int) -> dict:
    """Quality facts about the scaled coding that a reader of the paper needs."""
    from .core import RESIDUAL
    ok = [c for c in codings if c.status == "ok" and c.label is not None]
    in_domain = [c for c in ok if c.label.in_domain]
    resid = [c for c in in_domain if c.label.category == RESIDUAL]
    cited = [v for c in codings for v in c.evidence_verified.values()]
    return {
        "schema_version": schema.version, "schema_fingerprint": schema.fingerprint(),
        "rounds": n_rounds, "n_units": len(codings),
        "n_coding_errors": len(codings) - len(ok),
        "error_rate": (len(codings) - len(ok)) / len(codings) if codings else None,
        "n_in_domain": len(in_domain), "n_residual": len(resid),
        "coverage_of_in_domain": (1 - len(resid) / len(in_domain)) if in_domain else None,
        "evidence_passages_cited": len(cited),
        "evidence_passages_not_found_in_unit": sum(1 for v in cited if not v),
        "models": sorted({c.model_name for c in codings}),
        "counts": summarize_codings(codings),
    }


def summarize_codings(codings: list) -> dict:
    """Counts by label for a quick look at X."""
    out: dict = {}
    for c in codings:
        out[c.flat] = out.get(c.flat, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


__all__ = ["AASLRun", "Settings", "RunResult", "summarize_codings", "Coding"]
