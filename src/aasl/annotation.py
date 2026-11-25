"""Human coding of samples (line 3 and line 13 of Algorithm 1).

``FileAnnotator`` runs the real workflow: it writes one CSV per coder plus the
codebook, waits for the coders, checks inter-coder reliability, writes the
units the coders disagree on for resolution, and returns the resolved (gold)
labels. Each wait raises ``AwaitingHumanInput``; re-running resumes.

Reliability is checked *before* disagreements are resolved: if the coders do
not agree well enough, resolving individual cases is wasted work, and the
codebook needs tightening first.

CSV files saved by Excel are read as-is: a byte-order mark and a ``;`` or tab
delimiter (common in non-English locales) are detected.
"""
from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .core import RESIDUAL, AwaitingHumanInput, Label, ReliabilityTooLow
from .reliability import reliability_report
from .schema import Schema

YES = {"yes", "y", "1", "true", "t"}
NO = {"no", "n", "0", "false", "f"}
SEP = ";"


@dataclass
class AnnotationResult:
    gold: dict                      # uid -> Label
    reliability: dict = field(default_factory=dict)
    n_disagreements: int = 0


class Annotator(ABC):
    @abstractmethod
    def annotate(self, units: list, schema: Schema, tag: str) -> AnnotationResult:
        """Return human labels for ``units`` coded under ``schema``."""


# ---------------------------------------------------------------- CSV helpers
def read_csv_rows(path: Path) -> list:
    """Read a CSV that may have come from Excel: BOM, ``,`` / ``;`` / tab delimiter,
    stray spaces around column names."""
    text = Path(path).read_text(encoding="utf-8-sig")
    header = text.split("\n", 1)[0]
    counts = {d: header.count(d) for d in (",", ";", "\t")}
    delim = max(counts, key=counts.get) if max(counts.values()) else ","
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delim)
    rows = []
    for row in reader:
        rows.append({(k or "").strip(): (v if v is not None else "") for k, v in row.items()})
    return rows


def _has_subs(schema: Schema) -> bool:
    return any(c.subcategories for c in schema.categories)


def _columns(schema: Schema) -> list:
    cols = ["uid", "text", "in_domain", "category"]
    if _has_subs(schema):
        cols.append("subcategory")
    return cols + [f"attr:{a.name}" for a in schema.attributes] + ["comment"]


def write_template(path: Path, units: list, schema: Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_columns(schema))
        w.writeheader()
        for u in units:
            w.writerow({"uid": u.uid, "text": u.text})


def write_labels(path: Path, units: list, labels: dict, schema: Schema) -> None:
    """Write filled-in labels in template format (used for tests and imports)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_columns(schema))
        w.writeheader()
        for u in units:
            w.writerow(label_to_row(u, labels[u.uid], schema))


def label_to_row(unit, label: Label, schema: Schema) -> dict:
    row = {"uid": unit.uid, "text": unit.text, "in_domain": "yes" if label.in_domain else "no",
           "category": label.category or ""}
    if _has_subs(schema):
        row["subcategory"] = label.subcategory or ""
    for a in schema.attributes:
        row[f"attr:{a.name}"] = SEP.join(label.attributes.get(a.name, []))
    return row


def parse_row(row: dict, schema: Schema, prefix: str = "") -> tuple:
    """Parse one CSV row into a Label. Returns (label or None, problems); a row
    with no ``in_domain`` value is 'not coded yet' and returns (None, [])."""
    problems = []
    uid = (row.get("uid") or "").strip()
    raw_dom = (row.get(prefix + "in_domain") or "").strip().lower()
    if raw_dom in YES:
        in_domain = True
    elif raw_dom in NO:
        in_domain = False
    elif raw_dom:
        return None, [f"{uid}: in_domain is {raw_dom!r}; use yes or no"]
    else:
        return None, []
    if not in_domain:
        return Label(in_domain=False), problems
    cat = (row.get(prefix + "category") or "").strip()
    allowed = schema.category_names + [RESIDUAL]
    if cat not in allowed:
        return None, [f"{uid}: category {cat!r} is not one of {allowed}"]
    attrs, sub = {}, None
    if cat != RESIDUAL:
        category = schema.category(cat)
        if category.subcategories:
            raw_sub = (row.get(prefix + "subcategory") or "").strip()
            if raw_sub:
                if raw_sub not in category.subcategory_names:
                    problems.append(f"{uid}: subcategory {raw_sub!r} not in "
                                    f"{category.subcategory_names}")
                else:
                    sub = raw_sub
        for a in schema.attributes_for(cat):
            raw = (row.get(f"{prefix}attr:{a.name}") or "").strip()
            values = [v.strip() for v in raw.split(SEP) if v.strip()]
            if a.kind == "single" and len(values) > 1:
                problems.append(f"{uid}: {a.name} takes one value, got {values}")
            if a.kind in ("single", "multi"):
                bad = [v for v in values if v not in a.options]
                if bad:
                    problems.append(f"{uid}: {a.name} values {bad} not in {a.options}")
            attrs[a.name] = values
    return Label(in_domain=True, category=cat, attributes=attrs, subcategory=sub), problems


def read_labels(path: Path, schema: Schema, prefix: str = "") -> tuple:
    """Read a coder CSV. Returns ({uid: Label}, uids not coded yet, problems)."""
    labels, missing, problems = {}, [], []
    for row in read_csv_rows(path):
        uid = (row.get("uid") or "").strip()
        if not uid:
            problems.append("a row has no uid (is the first column named 'uid'?)")
            continue
        label, probs = parse_row(row, schema, prefix)
        problems += probs
        if label is None and not probs:
            missing.append(uid)
        elif label is not None:
            labels[uid] = label
    return labels, missing, problems


# ------------------------------------------------------------ file annotator
class FileAnnotator(Annotator):
    """Coders fill CSV files; disagreements are resolved in a separate file."""

    def __init__(self, root, n_coders: int = 2, min_reliability: Optional[float] = None,
                 reliability_metric: str = "alpha_nominal", on_low_reliability: str = "stop"):
        self.root = Path(root)
        self.n_coders = n_coders
        self.min_reliability = min_reliability
        self.reliability_metric = reliability_metric
        self.on_low_reliability = on_low_reliability

    def annotate(self, units: list, schema: Schema, tag: str) -> AnnotationResult:
        folder = self.root / tag
        coder_files = [folder / f"coder_{i + 1}.csv" for i in range(self.n_coders)]
        if not all(p.exists() for p in coder_files):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "codebook.md").write_text(schema.to_markdown(), encoding="utf-8")
            for p in coder_files:
                if not p.exists():
                    write_template(p, units, schema)
            (folder / "INSTRUCTIONS.txt").write_text(_INSTRUCTIONS, encoding="utf-8")
            raise AwaitingHumanInput(
                f"Human coding needed for '{tag}': {len(units)} units, {self.n_coders} coder file(s). "
                "Each coder works independently from codebook.md, then re-run.", coder_files)

        per_coder, problems = [], []
        uids = [u.uid for u in units]
        wanted = set(uids)
        for p in coder_files:
            labels, missing, probs = read_labels(p, schema)
            problems += [f"{p.name}: {x}" for x in probs]
            unknown = [u for u in list(labels) + missing if u not in wanted]
            if unknown:
                problems.append(f"{p.name}: {len(unknown)} row id(s) are not in this sample, e.g. "
                                f"{unknown[:3]} (did a spreadsheet program change the ids?)")
            absent = [u for u in uids if u not in labels]
            if absent:
                problems.append(f"{p.name}: {len(absent)} units not coded yet")
            per_coder.append(labels)
        if problems:
            report = folder / "PROBLEMS.txt"
            report.write_text("\n".join(problems) + "\n", encoding="utf-8")
            raise AwaitingHumanInput(f"Coder files for '{tag}' are incomplete or invalid; "
                                     f"see {report}", [report])

        flat = [[c[u].flat for u in uids] for c in per_coder]
        reliability = reliability_report(flat) if self.n_coders > 1 else {"n_coders": 1}
        disagree = [i for i, u in enumerate(uids) if len({f[i] for f in flat}) > 1]
        self._gate_reliability(folder, tag, reliability, units, flat, disagree)
        gold = {u: per_coder[0][u] for u in uids}
        if disagree:
            gold.update(self._resolve(folder, units, per_coder, disagree, schema, tag))
        return AnnotationResult(gold=gold, reliability=reliability, n_disagreements=len(disagree))

    def _gate_reliability(self, folder: Path, tag: str, reliability: dict, units: list,
                          flat: list, disagree: list) -> None:
        """Stop before resolution when coders agree too little to trust the sample."""
        value = reliability.get(self.reliability_metric)
        if self.min_reliability is None or value is None or value >= self.min_reliability:
            return
        if self.on_low_reliability != "stop":
            return
        path = folder / "RELIABILITY.txt"
        lines = [f"{self.reliability_metric} = {value:.3f}, below the minimum "
                 f"{self.min_reliability}.", "",
                 "Coders disagree on these units. Read them together, tighten codebook.md, and "
                 "have each coder re-code independently (edit the coder files), then re-run.", ""]
        for i in disagree[:50]:
            lines.append(f"{units[i].uid}: " + " | ".join(f[i] for f in flat) + f"  :: {units[i].text[:90]}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise ReliabilityTooLow(f"{tag}: {self.reliability_metric} = {value:.3f} < "
                                f"{self.min_reliability}; see {path}. No disagreements were "
                                "sent for resolution.")

    def _resolve(self, folder: Path, units: list, per_coder: list, idx: list,
                 schema: Schema, tag: str) -> dict:
        path = folder / "resolution.csv"
        rows = [units[i] for i in idx]
        disputed = {u.uid for u in rows}
        carried = {}
        if path.exists():
            old = read_csv_rows(path)
            if {r.get("uid") for r in old} == disputed:
                carried_ok = self._read_resolution(path, schema, rows)
                if carried_ok is not None:
                    return carried_ok
                raise AwaitingHumanInput(
                    f"resolution.csv for '{tag}' is incomplete or invalid; finish the final:* "
                    "columns for every row.", [path])
            # The coders changed their files after the list was written: keep the
            # finished decisions for units still in dispute and ask for the rest.
            carried = {r["uid"]: r for r in old if r.get("uid") in disputed}
            backup = path.with_name("resolution.stale.csv")
            path.replace(backup)
        cols = ["uid", "text"] + [f"coder_{k + 1}" for k in range(len(per_coder))] + \
            ["final:in_domain", "final:category"] + \
            (["final:subcategory"] if _has_subs(schema) else []) + \
            [f"final:attr:{a.name}" for a in schema.attributes] + ["rationale"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for u in rows:
                rec = {"uid": u.uid, "text": u.text}
                for k, c in enumerate(per_coder):
                    rec[f"coder_{k + 1}"] = c[u.uid].flat
                prev = carried.get(u.uid)
                if prev:
                    rec.update({c: prev.get(c, "") for c in cols if c.startswith("final:")
                                or c == "rationale"})
                w.writerow(rec)
        done = self._read_resolution(path, schema, rows)
        if done is not None:
            return done
        raise AwaitingHumanInput(
            f"{len(rows)} disagreements in '{tag}' need a resolved label in the final:* "
            "columns, then re-run.", [path])

    def _read_resolution(self, path: Path, schema: Schema, rows: list) -> Optional[dict]:
        resolved, missing, problems = read_labels(path, schema, prefix="final:")
        if missing or problems or len(resolved) < len(rows):
            return None
        return resolved


_INSTRUCTIONS = """Code every row of your coder_N.csv file independently, using codebook.md.

in_domain   : yes or no
category    : one category name exactly as written in the codebook, or RESIDUAL when the
              unit is in the domain but no category fits. Leave blank when in_domain is no.
subcategory : (only if the codebook lists subcategories) one of the category's
              subcategories, or blank if none fits.
attr:*      : attribute values; separate several values with a semicolon (;).
              Leave blank when the unit does not state the attribute.
comment     : optional.

Do not consult the other coder or any model output. Do not edit the uid column.
Save as CSV (UTF-8 or Excel's "CSV UTF-8"; comma, semicolon or tab delimiters are all
read) and re-run. Excel workbooks (.xlsx) are not read: use Save As > CSV.
"""
