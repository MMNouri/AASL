"""Evaluation against human labels and the threshold test of Algorithm 1.

The threshold is applied to precision and recall of *each* category and of
in-domain detection, not to a pooled average: a pooled score can pass while
a rare but important category fails. Categories with too few human-coded
units cannot be certified and count as failures, which tells the researcher
to enlarge or re-stratify the sample.

Scores are one-vs-rest over all units in the sample, so a unit the model sends
to the wrong category, to the residual, out of domain, or to an error counts
as a miss for its true category.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from .core import CODING_ERROR, OUT_OF_DOMAIN, RESIDUAL, normalize_text
from .schema import Schema


@dataclass
class Score:
    label: str
    support: int          # human-coded units with this label
    predicted: int        # model-coded units with this label
    tp: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    ci_precision: Optional[list] = None
    ci_recall: Optional[list] = None


def _prf(tp: int, predicted: int, support: int):
    p = tp / predicted if predicted else (None if support == 0 else 0.0)
    r = tp / support if support else None
    if p is None or r is None:
        f = None
    else:
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def _score(label: str, golds: list, preds: list) -> Score:
    tp = sum(1 for g, p in zip(golds, preds) if g and p)
    predicted = sum(1 for p in preds if p)
    support = sum(1 for g in golds if g)
    p, r, f = _prf(tp, predicted, support)
    return Score(label, support, predicted, tp, p, r, f)


@dataclass
class EvaluationReport:
    tau: float
    min_support: int
    n_units: int
    n_errors: int
    detection: Score
    categories: dict
    residual: Score
    attributes: dict
    gold_coverage: Optional[float]
    model_coverage: Optional[float]
    passes: bool
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    subcategories: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self, title: str = "Evaluation") -> str:
        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"

        def ci(c):
            return "" if not c else f" [{c[0]:.2f}, {c[1]:.2f}]"

        rows = [f"## {title}", "",
                f"Threshold tau = {self.tau}; min support = {self.min_support}; "
                f"units = {self.n_units}; coding errors = {self.n_errors}", "",
                "| label | support | predicted | precision | recall | F1 |",
                "|---|---:|---:|---:|---:|---:|"]
        for s in [self.detection, *self.categories.values(), self.residual]:
            rows.append(f"| {s.label} | {s.support} | {s.predicted} | {fmt(s.precision)}"
                        f"{ci(s.ci_precision)} | {fmt(s.recall)}{ci(s.ci_recall)} | {fmt(s.f1)} |")
        if self.subcategories:
            rows += ["", "Subcategories (scored on units where human and model agree on the "
                     "category):", "", "| subcategory | gold | predicted | precision | recall |",
                     "|---|---:|---:|---:|---:|"]
            for s in self.subcategories.values():
                rows.append(f"| {s.label} | {s.support} | {s.predicted} | {fmt(s.precision)} | "
                            f"{fmt(s.recall)} |")
        if self.attributes:
            rows += ["", "| attribute | gold values | predicted values | precision | recall |",
                     "|---|---:|---:|---:|---:|"]
            for s in self.attributes.values():
                rows.append(f"| {s.label} | {s.support} | {s.predicted} | {fmt(s.precision)} | "
                            f"{fmt(s.recall)} |")
        rows += ["", f"Schema coverage of in-domain units: human {fmt(self.gold_coverage)}, "
                 f"model {fmt(self.model_coverage)}", "",
                 f"**Result: {'PASS' if self.passes else 'FAIL'}**"]
        rows += [f"- failure: {f}" for f in self.failures]
        rows += [f"- warning: {w}" for w in self.warnings]
        return "\n".join(rows) + "\n"


def evaluate(gold: dict, codings: list, schema: Schema, tau: float, min_support: int = 10,
             gate_attributes: bool = False, gate_residual: bool = False,
             gate_subcategories: bool = False,
             cluster_of: Optional[dict] = None, n_boot: int = 0, seed: int = 0) -> EvaluationReport:
    """Score model ``codings`` against human ``gold`` labels ({uid: Label})."""
    by_uid = {c.uid: c for c in codings}
    uids = sorted(gold)
    gflat = [gold[u].flat for u in uids]
    pflat = [by_uid[u].flat if u in by_uid else CODING_ERROR for u in uids]
    n_errors = sum(1 for p in pflat if p == CODING_ERROR)

    def indicator(label: str, flats: list) -> list:
        return [f == label for f in flats]

    g_in = [f != OUT_OF_DOMAIN for f in gflat]
    p_in = [f not in (OUT_OF_DOMAIN, CODING_ERROR) for f in pflat]
    detection = _score("in-domain (detection)", g_in, p_in)

    categories = {c: _score(c, indicator(c, gflat), indicator(c, pflat))
                  for c in schema.category_names}
    residual = _score(RESIDUAL, indicator(RESIDUAL, gflat), indicator(RESIDUAL, pflat))

    attributes = _attribute_scores(gold, by_uid, schema, uids)
    subcategories = _subcategory_scores(gold, by_uid, schema, uids)

    if n_boot:
        clusters = [(cluster_of or {}).get(u, u) for u in uids]
        named = {"detection": (detection, g_in, p_in), RESIDUAL: (residual, indicator(RESIDUAL, gflat),
                 indicator(RESIDUAL, pflat))}
        for c in schema.category_names:
            named[c] = (categories[c], indicator(c, gflat), indicator(c, pflat))
        _bootstrap(named, clusters, n_boot, seed)

    failures, warnings = [], []
    gated = [detection] + list(categories.values()) + ([residual] if gate_residual else [])
    for s in gated:
        if s.support < min_support:
            failures.append(f"{s.label}: only {s.support} human-coded units (< {min_support}); "
                            "cannot certify, enlarge or re-stratify the sample")
            continue
        if s.precision is None or s.precision < tau:
            failures.append(f"{s.label}: precision {_f(s.precision)} < {tau}")
        if s.recall is None or s.recall < tau:
            failures.append(f"{s.label}: recall {_f(s.recall)} < {tau}")
    for s in attributes.values():
        if s.support < min_support:
            warnings.append(f"attribute {s.label}: only {s.support} gold values; not gated")
            continue
        bad = [m for m, v in (("precision", s.precision), ("recall", s.recall))
               if v is None or v < tau]
        if bad:
            msg = f"attribute {s.label}: " + ", ".join(
                f"{m} {_f(getattr(s, m))} < {tau}" for m in bad)
            (failures if gate_attributes else warnings).append(msg)
    for s in subcategories.values():
        if s.support < min_support:
            warnings.append(f"subcategory {s.label}: only {s.support} gold units; not gated")
            continue
        bad = [m for m, v in (("precision", s.precision), ("recall", s.recall))
               if v is None or v < tau]
        if bad:
            msg = f"subcategory {s.label}: " + ", ".join(
                f"{m} {_f(getattr(s, m))} < {tau}" for m in bad)
            (failures if gate_subcategories else warnings).append(msg)
    if n_errors:
        warnings.append(f"{n_errors} units had coding errors and were counted as misses")

    g_dom = [f for f in gflat if f != OUT_OF_DOMAIN]
    p_dom = [f for f in pflat if f not in (OUT_OF_DOMAIN, CODING_ERROR)]
    return EvaluationReport(
        tau=tau, min_support=min_support, n_units=len(uids), n_errors=n_errors,
        detection=detection, categories=categories, residual=residual, attributes=attributes,
        gold_coverage=coverage(g_dom), model_coverage=coverage(p_dom),
        passes=not failures, failures=failures, warnings=warnings,
        subcategories=subcategories)


def support_shortfalls(gold: dict, schema: Schema, min_support: int) -> list:
    """Gated labels that the *human* labels cannot certify (fewer than
    ``min_support`` units). Depends on the gold labels alone, so it can be
    checked before any model coding, and without looking at model performance.

    Each entry: label, support, n_units, needed (estimated sample size that
    would give ``min_support`` units at the observed rate, or None if the label
    was not seen at all).
    """
    n = len(gold)
    counts = {"in-domain (detection)": sum(1 for g in gold.values() if g.in_domain)}
    for c in schema.category_names:
        counts[c] = sum(1 for g in gold.values() if g.in_domain and g.category == c)
    out = []
    for label, sup in counts.items():
        if sup < min_support:
            needed = math.ceil(min_support / (sup / n) * 1.2) if sup and n else None
            out.append({"label": label, "support": sup, "n_units": n, "needed": needed})
    return out


def coverage(in_domain_flats: list) -> Optional[float]:
    """Seed-schema coverage: share of in-domain units that a schema category
    classifies (the rest are residual)."""
    if not in_domain_flats:
        return None
    return sum(1 for f in in_domain_flats if f != RESIDUAL) / len(in_domain_flats)


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _attribute_scores(gold: dict, by_uid: dict, schema: Schema, uids: list) -> dict:
    """Micro precision/recall over (unit, value) pairs, on units where human and
    model agree on the category, so attribute quality is not confounded with
    category errors."""
    out = {}
    for attr in schema.attributes:
        tp = pred = sup = 0
        for u in uids:
            g, c = gold[u], by_uid.get(u)
            if c is None or c.label is None or g.flat != c.label.flat:
                continue
            if not g.in_domain or g.category == RESIDUAL or not attr.applies(g.category):
                continue
            gv = {normalize_text(v) for v in g.attributes.get(attr.name, [])}
            pv = {normalize_text(v) for v in c.label.attributes.get(attr.name, [])}
            tp += len(gv & pv)
            pred += len(pv)
            sup += len(gv)
        p, r, f = _prf(tp, pred, sup)
        out[attr.name] = Score(attr.name, sup, pred, tp, p, r, f)
    return out


def _subcategory_scores(gold: dict, by_uid: dict, schema: Schema, uids: list) -> dict:
    """One-vs-rest precision/recall for each subcategory, on units where human and
    model agree on the category (so subcategory quality is not confounded with
    category errors). A unit with no subcategory counts as 'none fits'."""
    out = {}
    for cat in schema.categories:
        if not cat.subcategories:
            continue
        pairs = []
        for u in uids:
            g, c = gold[u], by_uid.get(u)
            if (c is None or c.label is None or not g.in_domain or g.category != cat.name
                    or c.label.category != cat.name):
                continue
            pairs.append((g.subcategory, c.label.subcategory))
        for sub in cat.subcategory_names:
            tp = sum(1 for g, p in pairs if g == sub and p == sub)
            pred = sum(1 for _, p in pairs if p == sub)
            sup = sum(1 for g, _ in pairs if g == sub)
            p_, r_, f_ = _prf(tp, pred, sup)
            out[f"{cat.name} / {sub}"] = Score(f"{cat.name} / {sub}", sup, pred, tp, p_, r_, f_)
    return out


def _bootstrap(named: dict, clusters: list, n_boot: int, seed: int) -> None:
    """Percentile CIs from resampling clusters (e.g. firms) with replacement."""
    ids = sorted(set(clusters), key=str)
    index = {c: i for i, c in enumerate(ids)}
    rng = random.Random(seed)
    for score, g, p in named.values():
        tp = [0] * len(ids)
        pr = [0] * len(ids)
        sp = [0] * len(ids)
        for cl, gi, pi in zip(clusters, g, p):
            k = index[cl]
            tp[k] += int(gi and pi)
            pr[k] += int(pi)
            sp[k] += int(gi)
        precs, recs = [], []
        for _ in range(n_boot):
            draw = [rng.randrange(len(ids)) for _ in ids]
            t = sum(tp[k] for k in draw)
            q = sum(pr[k] for k in draw)
            s = sum(sp[k] for k in draw)
            if q:
                precs.append(t / q)
            if s:
                recs.append(t / s)
        score.ci_precision = _pct(precs)
        score.ci_recall = _pct(recs)


def _pct(values: list) -> Optional[list]:
    if len(values) < 20:
        return None
    v = sorted(values)
    lo = v[int(0.025 * (len(v) - 1))]
    hi = v[int(math.ceil(0.975 * (len(v) - 1)))]
    return [round(lo, 4), round(hi, 4)]


def disagreements(gold: dict, codings: list, units: dict) -> list:
    """Rows where model and human differ, for the researcher revising prompts."""
    by_uid = {c.uid: c for c in codings}
    rows = []
    for u in sorted(gold):
        c = by_uid.get(u)
        pred = c.flat if c else CODING_ERROR
        if gold[u].flat != pred:
            rows.append({"uid": u, "human": gold[u].flat, "model": pred,
                         "model_evidence": (c.evidence.get("assign") or c.evidence.get("detect"))
                         if c else None,
                         "error": c.error if c else "not coded",
                         "text": units[u].text if u in units else ""})
    return rows


def required_sample_size(p: float = 0.9, margin: float = 0.05, z: float = 1.96,
                         design_effect: float = 1.0) -> int:
    """Units of a category needed so a proportion near ``p`` is estimated within
    +/- ``margin`` (normal approximation), inflated by the design effect of
    cluster sampling. Use it per category for both precision and recall."""
    return math.ceil(z * z * p * (1 - p) / (margin * margin) * design_effect)
