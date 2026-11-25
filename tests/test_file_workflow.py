"""The real workflow: humans work in files, the run pauses and resumes.

A scripted 'researcher' completes each file the run asks for, then re-runs.
"""
import csv
from pathlib import Path

import yaml

from aasl.adjudication import FileAdjudicator
from aasl.annotation import FileAnnotator, label_to_row, write_labels
from aasl.calibration import FileReviser
from aasl.core import AwaitingHumanInput
from aasl.pipeline import AASLRun, Settings
from aasl.schema import Schema
from aasl.simulate import (CLARIFY_TOKEN, HIDDEN_KEEP, SimulatedModel, _u, make_corpus, project,
                           seed_schema)
from aasl.store import RunStore


def researcher(exc, run_dir: Path, units: dict, truth: dict, log: list):
    schema = Schema.from_dict(RunStore(run_dir).load("schema_current"))
    for p in map(Path, exc.paths):
        log.append(p.name)
        if p.name.startswith("coder_"):
            folder = p.parent
            rows = list(csv.DictReader(open(folder / "coder_1.csv", encoding="utf-8")))
            subset = [units[r["uid"]] for r in rows]
            labels = {u.uid: project(truth[u.uid], schema) for u in subset}
            write_labels(folder / "coder_1.csv", subset, labels, schema)
            # coder 2 codes the same truth but slips on ~2% of units
            slipped = {u: (project(truth[u], schema) if _u(u, "c2") > 0.02 else
                           project(truth[u].__class__("OUT_OF_DOMAIN"), schema)) for u in labels}
            write_labels(folder / "coder_2.csv", subset, slipped, schema)
        elif p.name == "resolution.csv":
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            cols = list(rows[0].keys())
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    lab = label_to_row(units[r["uid"]], project(truth[r["uid"]], schema), schema)
                    r.update({f"final:{k}": v for k, v in lab.items() if k not in ("uid", "text")})
                    r["rationale"] = "checked against codebook"
                    w.writerow({k: r.get(k, "") for k in cols})
        elif p.name == "revision.yaml":
            doc = yaml.safe_load(p.read_text())
            doc["notes"].setdefault("Joint development", []).append(
                f"Joint development means {CLARIFY_TOKEN}; sharing existing assets is Resource pooling.")
            doc["reason"] = "model sent co-design to Resource pooling"
            doc["ready"] = True
            p.write_text(yaml.safe_dump(doc))
        elif p.name == "proposals.yaml":
            doc = yaml.safe_load(p.read_text())
            for item in doc["proposals"]:
                keep = item["name"] == HIDDEN_KEEP
                item["decision"].update(action="keep" if keep else "reject",
                                        rationale="theory: collective rule-making" if keep
                                        else "no joint activity")
            doc["ready"] = True
            p.write_text(yaml.safe_dump(doc))
        else:
            raise AssertionError(f"unexpected request: {p}")


def test_file_based_run_pauses_and_resumes(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)
    by_uid = {u.uid: u for u in units}
    settings = Settings(tau=0.9, min_support=15, holdout_size=400, calibration_size=400,
                        strata=["sector", "region"], cluster="firm", max_per_cluster=6, seed=7,
                        n_boot=100, backoff=0.0, max_workers=4)
    model = SimulatedModel(truth)
    requests = []
    for attempt in range(30):
        run = AASLRun(units, seed_schema(), model,
                      FileAnnotator(tmp_path / "human" / "coding", n_coders=2),
                      FileReviser(tmp_path / "human" / "revisions"),
                      FileAdjudicator(tmp_path / "human" / "adjudication"),
                      settings, tmp_path, verbose=False)
        try:
            result = run.run()
            break
        except AwaitingHumanInput as exc:
            researcher(exc, tmp_path, by_uid, truth, requests)
    else:
        raise AssertionError("run never finished")

    assert result.schema.category_names[-1] == HIDDEN_KEEP
    assert result.holdout_report.passes
    # every kind of human step was requested, in order
    assert requests.index("revision.yaml") < requests.index("proposals.yaml")
    assert "resolution.csv" in requests
    # reliability was computed from the two coder files and stored with the gold labels
    gold = RunStore(tmp_path).load("gold__round1_calibration")
    assert 0.9 < gold["reliability"]["cohen_kappa"] < 1.0
    assert (tmp_path / "human" / "coding" / "holdout" / "codebook.md").exists()
