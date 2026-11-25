"""Simulation for testing AASL without a provider or human coders.

The synthetic corpus has a known truth, including categories that are *not*
in the seed schema. A correct implementation of Algorithm 1 must:

* need at least one prompt revision in round 1 (the simulated model confuses
  two seed categories until the prompt clarifies the difference);
* route units of unseeded categories to the residual;
* receive a proposal for each unseeded category from the residual, plus a
  fabricated proposal whose quotes do not exist, which verification must drop;
* add the category the researcher keeps, reject the other, and stop when a
  round keeps nothing;
* pass held-out confirmation and code every unit with the final schema.

None of this is used by the real workflow; it exists for tests and the demo.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from dataclasses import dataclass, field

from .adjudication import Adjudicator, Decision
from .annotation import AnnotationResult, Annotator
from .calibration import PromptReviser
from .core import NO_SUBCATEGORY, OUT_OF_DOMAIN, RESIDUAL, Label, TextUnit
from .llm.base import LanguageModel
from .prompts import NOT_STATED
from .reliability import reliability_report
from .schema import Attribute, Category, Domain, Schema, Subcategory

SEED_CATEGORIES = ["Knowledge exchange", "Resource pooling", "Joint development", "Joint promotion"]
HIDDEN_KEEP = "Standard setting"
HIDDEN_REJECT = "Philanthropic partnership"
CLARIFY_TOKEN = "jointly creating something new"

_TEMPLATES = {
    "Knowledge exchange": [
        "{firm} and {partner} ran a series of workshops to share expertise on {topic}{gov}.",
        "{firm} exchanged best practice on {topic} with {partner}{gov}.",
        "Engineers from {firm} and {partner} trained each other on {topic} methods{gov}.",
    ],
    "Resource pooling": [
        "{firm} pooled funding with {partner} to finance shared {topic} infrastructure{gov}.",
        "{firm} opened its existing {topic} facilities to {partner} for joint use{gov}.",
        "{firm} and {partner} shared a logistics fleet to cut {topic} costs{gov}.",
    ],
    "Joint development": [
        "{firm} and {partner} co-designed a new {topic} product{gov}.",
        "Together with {partner}, {firm} built a new {topic} solution from scratch{gov}.",
        "{firm} worked with {partner} to develop a novel {topic} process{gov}.",
    ],
    "Joint promotion": [
        "{firm} and {partner} launched a joint campaign promoting {topic} to consumers{gov}.",
        "{firm} co-branded a {topic} awareness drive with {partner}{gov}.",
    ],
    HIDDEN_KEEP: [
        "{firm} joined {partner} and peer companies to draft an industry standard for {topic}{gov}.",
        "{firm} sat on a working group with {partner} that wrote certification criteria for {topic}{gov}.",
    ],
    HIDDEN_REJECT: [
        "{firm} teamed up with {partner} to donate {topic} kits to local schools{gov}.",
        "Through {partner}, {firm} gave grants to community {topic} projects{gov}.",
    ],
    OUT_OF_DOMAIN: [
        "{firm} cut its {topic} footprint by {pct}% through internal efficiency measures.",
        "{firm} trained all employees on {topic} awareness.",
        "{firm} installed new {topic} monitoring equipment at its plants.",
        "{firm} published its first report on {topic} performance.",
    ],
}

_WEIGHTS = {OUT_OF_DOMAIN: 0.40, "Knowledge exchange": 0.10, "Resource pooling": 0.10,
            "Joint development": 0.12, "Joint promotion": 0.08, HIDDEN_KEEP: 0.10,
            HIDDEN_REJECT: 0.10}

_PARTNERS = [("Green Earth Alliance", "NGO"), ("Ocean Care Trust", "NGO"),
             ("Ministry of Environment", "Government"), ("City of Northport", "Government"),
             ("Northfield University", "Research institution"),
             ("Delta Institute of Technology", "Research institution"),
             ("Atlas Logistics", "Business"), ("Kestrel Materials", "Business"),
             ("Global Packaging Council", "Industry association")]
_TOPICS = ["water", "energy", "packaging", "recycling", "emissions", "biodiversity", "waste"]
_GOV = [("", []), (" under a memorandum of understanding", ["Formal agreement"]),
        (" through a jointly owned venture", ["Equity joint venture"])]
_DEFS = {HIDDEN_KEEP: "Organisations jointly write or govern standards, codes or certification "
                      "rules for a sector.",
         HIDDEN_REJECT: "The firm channels donations or grants through a partner without "
                        "joint activity."}


# Knowledge-exchange templates 0 and 2 describe training, template 1 best-practice exchange.
_KE_SUB = {0: "Training", 1: "Practice exchange", 2: "Training"}


@dataclass
class Truth:
    category: str
    attributes: dict = field(default_factory=dict)
    subcategory: str = None


def seed_schema() -> Schema:
    """Seed schema S for the synthetic domain (four seeds, three attributes)."""
    return Schema(
        domain=Domain(
            name="Inter-organisational cooperation",
            definition="The firm acts together with at least one external organisation toward a "
                       "shared goal.",
            include=["joint activity with a named or described external party"],
            exclude=["actions the firm takes alone", "internal programmes for its own staff"]),
        categories=[
            Category("Knowledge exchange", "Partners share expertise, information or training.",
                     source="synthetic seed",
                     subcategories=[Subcategory("Training", "Partners teach or train each other."),
                                    Subcategory("Practice exchange",
                                                "Partners swap existing practices or know-how.")]),
            Category("Resource pooling", "Partners combine or share money, assets or facilities.",
                     source="synthetic seed"),
            Category("Joint development", "Partners develop a product, process or solution "
                     "together.", source="synthetic seed"),
            Category("Joint promotion", "Partners promote a cause, product or message together.",
                     source="synthetic seed"),
        ],
        attributes=[
            Attribute("governance_form", "Formal basis of the cooperation", "single",
                      ["Equity joint venture", "Formal agreement"]),
            Attribute("partner_type", "Type of each external partner", "multi",
                      sorted({t for _, t in _PARTNERS})),
            Attribute("partner_names", "Names of external partners", "text"),
        ])


def make_corpus(n: int = 3000, n_firms: int = 120, seed: int = 11) -> tuple:
    """Return (units, truth by uid)."""
    rng = random.Random(seed)
    sectors = ["Energy", "Consumer goods", "Industrials", "Financials"]
    regions = ["Europe", "North America", "Asia"]
    firms = [{"firm": f"Firm{i:03d}", "sector": sectors[i % 4], "region": regions[i % 3]}
             for i in range(n_firms)]
    cats, weights = list(_WEIGHTS), list(_WEIGHTS.values())
    units, truth = [], {}
    for i in range(n):
        f = rng.choice(firms)
        cat = rng.choices(cats, weights)[0]
        t_idx = rng.randrange(len(_TEMPLATES[cat]))
        template = _TEMPLATES[cat][t_idx]
        partner, ptype = rng.choice(_PARTNERS)
        gov_text, gov_val = rng.choice(_GOV) if cat != OUT_OF_DOMAIN else ("", [])
        text = template.format(firm=f["firm"], partner=partner, topic=rng.choice(_TOPICS),
                               gov=gov_text, pct=rng.randint(5, 30))
        uid = f"u{i:05d}"
        units.append(TextUnit(uid, text, {"firm": f["firm"], "sector": f["sector"],
                                          "region": f["region"],
                                          "year": str(rng.randint(2015, 2022))}))
        attrs = {} if cat == OUT_OF_DOMAIN else {
            "governance_form": gov_val, "partner_type": [ptype], "partner_names": [partner]}
        truth[uid] = Truth(cat, attrs, _KE_SUB[t_idx] if cat == "Knowledge exchange" else None)
    return units, truth


def project(truth: Truth, schema: Schema) -> Label:
    """What a careful human coder records under ``schema``."""
    if truth.category == OUT_OF_DOMAIN:
        return Label(in_domain=False)
    if truth.category not in schema.category_names:
        return Label(in_domain=True, category=RESIDUAL)
    attrs = {a.name: list(truth.attributes.get(a.name, []))
             for a in schema.attributes_for(truth.category)}
    sub = truth.subcategory if schema.category(truth.category).subcategories else None
    return Label(in_domain=True, category=truth.category, attributes=attrs, subcategory=sub)


def _u(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# ------------------------------------------------------------------- model
class SimulatedModel(LanguageModel):
    """Answers from the known truth with controlled imperfections."""

    _UNIT = re.compile(r'<unit id="([^"]+)">')

    def __init__(self, truth: dict, detect_noise: float = 0.01, confusion: float = 0.35,
                 malformed_rate: float = 0.01, fabricate: bool = True):
        self.name = "simulated"
        self.truth = truth
        self.detect_noise = detect_noise
        self.confusion = confusion
        self.malformed_rate = malformed_rate
        self.fabricate = fabricate
        self.calls = 0
        self._first_try = set()
        self._lock = threading.Lock()

    def generate(self, system: str, user: str, schema: dict, task: str) -> str:
        with self._lock:
            self.calls += 1
        ids = self._UNIT.findall(user)
        if task == "propose":
            return json.dumps(self._propose(ids, user))
        uid = ids[0]
        key = (uid, task)
        if _u(uid, task, "malformed") < self.malformed_rate and "previous reply was rejected" not in user:
            with self._lock:
                if key not in self._first_try:
                    self._first_try.add(key)
                    return '{"broken": '          # invalid JSON: must be retried
        t = self.truth[uid]
        if task == "detect":
            in_dom = t.category != OUT_OF_DOMAIN
            if _u(uid, "detect") < self.detect_noise:
                in_dom = not in_dom
            return json.dumps({"in_domain": in_dom, "evidence": " ".join(self._text(user).split()[:6])})
        if task == "assign":
            allowed = schema["properties"]["category"]["enum"]
            cat = t.category if t.category in allowed else RESIDUAL
            clarified = CLARIFY_TOKEN in user
            if (cat == "Joint development" and "Resource pooling" in allowed and not clarified
                    and _u(uid, "confuse") < self.confusion):
                cat = "Resource pooling"
            return json.dumps({"category": cat, "evidence": " ".join(self._text(user).split()[:8])})
        if task == "subcategory":
            allowed = schema["properties"]["subcategory"]["enum"]
            return json.dumps({"subcategory": t.subcategory if t.subcategory in allowed
                               else NO_SUBCATEGORY})
        if task == "attributes":
            out = {}
            for name, spec in schema["properties"].items():
                vals = t.attributes.get(name, [])
                if spec["type"] == "string":
                    out[name] = vals[0] if vals else NOT_STATED
                else:
                    out[name] = list(vals)
            return json.dumps(out)
        raise ValueError(f"unknown task {task}")

    def _text(self, user: str) -> str:
        m = re.search(r'<unit id="[^"]+">\n(.*?)\n</unit>', user, re.S)
        return m.group(1) if m else ""

    def _propose(self, ids: list, user: str) -> dict:
        texts = dict(re.findall(r'<unit id="([^"]+)">\n(.*?)\n</unit>', user, re.S))
        groups: dict = {}
        for uid in ids:
            cat = self.truth[uid].category
            if cat in (OUT_OF_DOMAIN,) or f"- {cat}:" in user:
                continue
            groups.setdefault(cat, []).append(uid)
        proposals = []
        for cat, members in sorted(groups.items()):
            if len(members) < 2 or cat not in _DEFS:
                continue
            proposals.append({"name": cat, "definition": _DEFS[cat], "supporting": [
                {"unit_id": u, "quote": " ".join(texts[u].split()[:7])} for u in members[:6]]})
        if self.fabricate and len(ids) >= 2:
            proposals.append({"name": "Climate lobbying", "definition": "Joint lobbying.",
                              "supporting": [{"unit_id": u, "quote": "we lobbied parliament together"}
                                             for u in ids[:3]]})
        return {"proposals": proposals}


# -------------------------------------------------------------- humans
class SimulatedAnnotator(Annotator):
    """Two coders who code the truth; the second slips on a small share of units."""

    def __init__(self, truth: dict, slip: float = 0.03):
        self.truth = truth
        self.slip = slip
        self.calls = []

    def annotate(self, units, schema, tag):
        self.calls.append(tag)
        gold = {u.uid: project(self.truth[u.uid], schema) for u in units}
        options = [OUT_OF_DOMAIN, RESIDUAL] + schema.category_names
        c1 = [gold[u.uid].flat for u in units]
        c2 = [(options[int(_u(u.uid, tag) * 1e6) % len(options)]
               if _u(u.uid, tag, "slip") < self.slip else f) for u, f in zip(units, c1)]
        rel = reliability_report([c1, c2])
        return AnnotationResult(gold=gold, reliability=rel,
                                n_disagreements=sum(a != b for a, b in zip(c1, c2)))


class ScriptedReviser(PromptReviser):
    """Adds the clarification a researcher would write after reading the errors."""

    def __init__(self):
        self.calls = 0

    def revise(self, schema, report, errors, tag):
        self.calls += 1
        notes = {k: list(v) for k, v in schema.notes.items()}
        jd = notes.setdefault("Joint development", [])
        note = (f"Joint development means {CLARIFY_TOKEN} (a product, process or solution). "
                "Sharing assets that already exist is Resource pooling.")
        if note not in jd:
            jd.append(note)
        else:
            notes.setdefault("categories", []).append(f"Revision {self.calls}: re-read definitions.")
        return schema.with_notes(notes, reason=f"{tag}: JD/RP confusion")


class ScriptedAdjudicator(Adjudicator):
    def __init__(self, keep: tuple = (HIDDEN_KEEP,)):
        self.keep = set(keep)
        self.seen = []

    def decide(self, proposals, schema, tag):
        out = []
        for p in proposals:
            self.seen.append((tag, p.name))
            if p.name in self.keep:
                out.append(Decision(p.pid, "keep", "Collective rule-making is a distinct form of "
                                    "cooperation not covered by the seeds.",
                                    source="synthetic: researcher judgment"))
            else:
                out.append(Decision(p.pid, "reject", "Outside the domain definition: no joint "
                                    "activity between the partners."))
        return out
