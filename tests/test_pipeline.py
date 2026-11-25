"""End-to-end tests of Algorithm 1 on the synthetic corpus."""
import json

import pytest

from aasl.core import RESIDUAL, ConfirmationFailed
from aasl.pipeline import AASLRun, Settings
from aasl.simulate import (HIDDEN_KEEP, HIDDEN_REJECT, ScriptedAdjudicator, ScriptedReviser,
                           SimulatedAnnotator, SimulatedModel, make_corpus, seed_schema)


def settings(**kw):
    base = dict(tau=0.9, min_support=15, holdout_size=400, calibration_size=400,
                strata=["sector", "region"], cluster="firm", max_per_cluster=6, seed=7,
                n_boot=200, backoff=0.0, max_workers=4)
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    units, truth = make_corpus(n=3000, seed=7)
    run_dir = tmp_path_factory.mktemp("run")
    parts = dict(model=SimulatedModel(truth), annotator=SimulatedAnnotator(truth),
                 reviser=ScriptedReviser(), adjudicator=ScriptedAdjudicator())
    run = AASLRun(units, seed_schema(), parts["model"], parts["annotator"], parts["reviser"],
                  parts["adjudicator"], settings(), run_dir, verbose=False)
    result = run.run()
    return dict(units=units, truth=truth, run=run, result=result, run_dir=run_dir, **parts)


def test_lines_1_to_12_rounds(finished):
    r = finished["result"]
    # round 1 discovers and keeps the unseeded category; round 2 keeps nothing and stops
    assert [x["kept"] for x in r.rounds] == [[HIDDEN_KEEP], []]
    assert r.schema.category_names[-1] == HIDDEN_KEEP
    cat = r.schema.category(HIDDEN_KEEP)
    assert cat.origin == "discovered" and cat.added_in_round == 1
    assert HIDDEN_REJECT not in r.schema.category_names


def test_line_6_prompts_revised_in_round_1(finished):
    r = finished["result"]
    assert r.rounds[0]["calibration_passes"] >= 2
    assert finished["reviser"].calls >= 1
    assert r.schema.notes_for("Joint development")
    # calibration passes are recorded with schema fingerprints
    fps = [it["fingerprint"] for it in r.rounds[0]["calibration"]]
    assert len(set(fps)) == len(fps)


def test_line_3_fresh_sample_each_round_and_holdout_untouched(finished):
    led = finished["run"].ledger
    hold = set(led.get("holdout"))
    a1, a2 = set(led.get("round1/calibration")), set(led.get("round2/calibration"))
    assert not (hold & a1) and not (hold & a2) and not (a1 & a2)
    run_dir = finished["run_dir"]
    for k in (1, 2):
        disc = json.loads((run_dir / "state" / f"round{k}__discovery.json").read_text())
        assert not (set(disc["residual_uids"]) & hold)
    assert finished["annotator"].calls == ["round1_calibration", "round2_calibration", "holdout"]


def test_lines_8_9_residual_proposals_and_verification(finished):
    r = finished["result"]
    first = r.rounds[0]
    assert first["n_residual"] > 0 and first["coverage"] < 1
    dropped = {d["name"] for d in first["dropped_proposals"]}
    assert "Climate lobbying" in dropped                  # fabricated quotes removed
    proposed = [name for tag, name in finished["adjudicator"].seen]
    # a rejected proposal is not put in front of the researcher again in round 2
    assert proposed.count(HIDDEN_REJECT) == 1 and proposed.count(HIDDEN_KEEP) == 1


def test_line_10_decisions_logged(finished):
    log = [json.loads(l) for l in (finished["run_dir"] / "decisions.jsonl").read_text().splitlines()]
    assert {(e["proposal"], e["action"]) for e in log} == {
        (HIDDEN_KEEP, "keep"), (HIDDEN_REJECT, "reject")}
    assert all(e["rationale"] for e in log)


def test_line_13_holdout_confirmation(finished):
    rep = finished["result"].holdout_report
    assert rep.passes
    assert HIDDEN_KEEP in rep.categories and rep.categories[HIDDEN_KEEP].support >= 15
    assert rep.categories[HIDDEN_KEEP].ci_recall is not None
    assert (finished["run_dir"] / "reports" / "holdout_confirmation.md").exists()


def test_line_14_all_units_coded_with_final_schema(finished):
    r = finished["result"]
    assert len(r.codings) == len(finished["units"])
    fp = r.schema.fingerprint()
    assert all(c.schema_fingerprint == fp for c in r.codings)
    truth = finished["truth"]
    resid = [c for c in r.codings if c.flat == RESIDUAL]
    share = sum(truth[c.uid].category == HIDDEN_REJECT for c in resid) / len(resid)
    assert share > 0.9                                    # what is left is the rejected kind
    out = finished["run_dir"] / "output"
    for name in ("final_schema.yaml", "codebook.md", "coded_units.csv", "coded_units.jsonl"):
        assert (out / name).exists()


def test_resume_is_free_and_identical(finished):
    units, truth = finished["units"], finished["truth"]
    model = SimulatedModel(truth)
    run = AASLRun(units, seed_schema(), model, SimulatedAnnotator(truth), ScriptedReviser(),
                  ScriptedAdjudicator(), settings(), finished["run_dir"], verbose=False)
    again = run.run()
    assert model.calls == 0
    assert again.schema.to_dict() == finished["result"].schema.to_dict()


def test_stops_after_one_round_when_nothing_kept(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)
    run = AASLRun(units, seed_schema(), SimulatedModel(truth), SimulatedAnnotator(truth),
                  ScriptedReviser(), ScriptedAdjudicator(keep=()), settings(), tmp_path,
                  verbose=False)
    r = run.run()
    assert len(r.rounds) == 1 and r.schema.category_names == seed_schema().category_names


def test_confirmation_failure_blocks_scaling(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)

    class DriftingModel(SimulatedModel):
        """Behaves during calibration, then degrades on the held-out units."""
        def __init__(self, truth, holdout):
            super().__init__(truth)
            self.holdout = holdout

        def generate(self, system, user, schema, task):
            if task == "assign" and any(f'id="{u}"' in user for u in self.holdout):
                return json.dumps({"category": "Joint promotion", "evidence": ""})
            return super().generate(system, user, schema, task)

    run = AASLRun(units, seed_schema(), SimulatedModel(truth), SimulatedAnnotator(truth),
                  ScriptedReviser(), ScriptedAdjudicator(), settings(), tmp_path, verbose=False)
    hold = {u.uid for u in run._draw("holdout", 400)}
    run.model = DriftingModel(truth, hold)
    with pytest.raises(ConfirmationFailed):
        run.run()
    assert not (tmp_path / "output" / "coded_units.csv").exists()


# ----------------------------------------------------------------------------
# Behaviour added after the review: support checks, extension, guards, audit.
# ----------------------------------------------------------------------------
import re

from aasl.core import AASLError, InsufficientSupport, ModelUnavailable, ReliabilityTooLow
from aasl.llm.base import LanguageModel


class SpyModel(SimulatedModel):
    """Records which units were coded and every discovery prompt."""

    def __init__(self, truth):
        super().__init__(truth)
        self.coded = set()
        self.propose_prompts = []

    def generate(self, system, user, schema, task):
        if task == "propose":
            self.propose_prompts.append(user)
        else:
            self.coded.update(re.findall(r'<unit id="([^"]+)">', user))
        return super().generate(system, user, schema, task)


def make_run(tmp_path, units, truth, model=None, annotator=None, reviser=None, **kw):
    model = model or SimulatedModel(truth)
    annotator = annotator or SimulatedAnnotator(truth)
    reviser = reviser or ScriptedReviser()
    run = AASLRun(units, seed_schema(), model, annotator, reviser, ScriptedAdjudicator(),
                  settings(n_boot=0, **kw), tmp_path, verbose=False)
    return run, model, annotator, reviser


def test_support_shortfall_stops_before_any_model_call_and_extension_keeps_coded_units(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)
    run, model, annot, rev = make_run(tmp_path, units, truth, min_support=25, calibration_size=250)
    with pytest.raises(InsufficientSupport) as exc:
        run.run()
    assert "Joint promotion" in str(exc.value) and "calibration_size" in str(exc.value)
    assert model.calls == 0 and rev.calls == 0              # no model coding, no revision request
    first = list(run.ledger.get("round1/calibration"))
    assert len(first) == 250

    # raise the size as the message says: the coded units stay, only new ones are added
    run2, model2, annot2, _ = make_run(tmp_path, units, truth, min_support=25, calibration_size=550)
    result = run2.run()
    parts = run2.ledger.parts("round1/calibration")
    assert parts == ["round1/calibration", "round1/calibration+1"]
    assert run2.ledger.get(parts[0]) == first
    assert [len(run2.ledger.samples[p]) for p in parts] == [250, 300]
    assert annot2.calls.count("round1_calibration") == 0        # not asked of the coders again
    assert annot2.calls.count("round1_calibration_ext1") == 1   # only the added units
    hold = run2.ledger.holdout_uids()
    everything = [set(run2.ledger.samples[p]) for p in parts]
    assert not (everything[0] & everything[1]) and not (hold & (everything[0] | everything[1]))
    firms = {}
    by_uid = {u.uid: u for u in units}
    for uid in everything[0] | everything[1]:
        f = by_uid[uid].meta["firm"]
        firms[f] = firms.get(f, 0) + 1
    assert max(firms.values()) <= 6                              # cluster cap holds across parts
    assert result.holdout_report.passes and result.schema.category_names[-1] == HIDDEN_KEEP


def test_holdout_support_is_checked_before_the_model_sees_the_holdout(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)
    spy = SpyModel(truth)
    run, *_ = make_run(tmp_path, units, truth, model=spy, min_support=25, calibration_size=550,
                       holdout_size=150)
    with pytest.raises(InsufficientSupport, match="holdout_size"):
        run.run()
    base = set(run.ledger.get("holdout"))
    assert len(base) == 150 and not (spy.coded & base)           # never coded by the model
    run2, *_ = make_run(tmp_path, units, truth, min_support=25, calibration_size=550,
                        holdout_size=500)
    result = run2.run()
    assert run2.ledger.parts("holdout") == ["holdout", "holdout+1"]
    others = {u for k, v in run2.ledger.samples.items() if not k.startswith("holdout") for u in v}
    assert not (run2.ledger.holdout_uids() & others) and len(run2.ledger.holdout_uids()) == 500
    assert result.holdout_report.passes and result.holdout_report.n_units == 500


def test_changing_the_seed_schema_or_corpus_mid_run_is_refused(finished):
    units, truth = finished["units"], finished["truth"]
    other = seed_schema()
    other.categories[0].definition = "A different idea entirely."
    kw = dict(model=SimulatedModel(truth), annotator=SimulatedAnnotator(truth),
              reviser=ScriptedReviser(), adjudicator=ScriptedAdjudicator(), settings=settings(),
              run_dir=finished["run_dir"], verbose=False)
    with pytest.raises(AASLError, match="seed schema changed"):
        AASLRun(units, other, **kw).run()
    with pytest.raises(AASLError, match="corpus changed"):
        AASLRun(units[:-1], seed_schema(), **kw).run()


def test_rejected_proposals_are_not_offered_again(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)
    spy = SpyModel(truth)
    run, *_ = make_run(tmp_path, units, truth, model=spy)
    run.run()
    first_round, later = spy.propose_prompts[0], spy.propose_prompts[-1]
    assert "already rejected" not in first_round
    assert "already rejected" in later and HIDDEN_REJECT in later
    assert HIDDEN_KEEP not in later.split("already rejected")[1].split("Propose new")[0]


class LowAlphaAnnotator(SimulatedAnnotator):
    def annotate(self, units, schema, tag):
        res = super().annotate(units, schema, tag)
        res.reliability = dict(res.reliability, alpha_nominal=0.3)
        return res


def test_reliability_below_minimum_stops_the_run(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)
    run, *_ = make_run(tmp_path, units, truth, annotator=LowAlphaAnnotator(truth))
    with pytest.raises(ReliabilityTooLow, match="alpha_nominal = 0.300"):
        run.run()
    assert not run.store.has("gold/round1_calibration")        # an untrusted sample is not kept
    events = (tmp_path / "events.jsonl").read_text()
    assert "reliability_too_low" in events


def test_reliability_gate_can_be_set_to_warn(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)
    run, *_ = make_run(tmp_path, units, truth, annotator=LowAlphaAnnotator(truth),
                       on_low_reliability="warn")
    assert run.run().holdout_report.passes
    assert "reliability_warning" in (tmp_path / "events.jsonl").read_text()


def test_a_dead_model_stops_the_pipeline(tmp_path):
    units, truth = make_corpus(n=3000, seed=7)

    class Dead(LanguageModel):
        name = "dead"

        def generate(self, system, user, schema, task):
            raise RuntimeError("503 service unavailable")

    run, *_ = make_run(tmp_path, units, truth, model=Dead(), retries=0)
    with pytest.raises(ModelUnavailable, match="503"):
        run.run()


def test_subcategories_are_coded_scored_and_exported(finished):
    import csv
    rep = finished["result"].holdout_report
    assert set(rep.subcategories) == {"Knowledge exchange / Training",
                                      "Knowledge exchange / Practice exchange"}
    assert all(s.support >= 5 and s.precision >= 0.9 for s in rep.subcategories.values())
    rows = list(csv.DictReader(open(finished["run_dir"] / "output" / "coded_units.csv",
                                    encoding="utf-8")))
    assert "subcategory" in rows[0]
    ke = [r for r in rows if r["category"] == "Knowledge exchange"]
    other = [r for r in rows if r["category"] not in ("Knowledge exchange", "")]
    assert ke and all(r["subcategory"] in ("Training", "Practice exchange") for r in ke)
    assert all(r["subcategory"] == "" for r in other)
    truth = finished["truth"]
    right = sum(r["subcategory"] == truth[r["uid"]].subcategory for r in ke
                if truth[r["uid"]].category == "Knowledge exchange")
    assert right / len(ke) > 0.95


def test_audit_file_reports_scaled_coding_quality(finished):
    audit = json.loads((finished["run_dir"] / "output" / "audit.json").read_text())
    codings = finished["result"].codings
    assert audit["n_units"] == 3000 == len(codings) and audit["n_coding_errors"] == 0
    in_dom = [c for c in codings if c.label.in_domain]
    resid = [c for c in in_dom if c.label.category == RESIDUAL]
    assert audit["n_in_domain"] == len(in_dom) and audit["n_residual"] == len(resid)
    assert audit["coverage_of_in_domain"] == pytest.approx(1 - len(resid) / len(in_dom))
    assert audit["evidence_passages_not_found_in_unit"] == 0 and audit["evidence_passages_cited"] > 0
    assert audit["schema_fingerprint"] == finished["result"].schema.fingerprint()
