import pytest
import yaml

from aasl.adjudication import Decision, FileAdjudicator, apply_decisions
from aasl.core import AwaitingHumanInput
from aasl.discovery import Evidence, Proposal


def props():
    return [Proposal("r1-p1", "Standard setting", "Writing shared rules.",
                     [Evidence("u1", "code of conduct", True), Evidence("u2", "standard", True)]),
            Proposal("r1-p2", "Rule making", "Drafting rules.", [Evidence("u3", "rules", True)]),
            Proposal("r1-p3", "Tech sharing", "Sharing technology.", [Evidence("u4", "tech", True)]),
            Proposal("r1-p4", "Donations", "Giving money.", [Evidence("u5", "grant", True)])]


def test_keep_merge_reject(small_schema):
    decisions = [Decision("r1-p1", "keep", "distinct form", name="Standard-setting",
                          source="theory ref"),
                 Decision("r1-p2", "merge", "same thing", into="r1-p1"),
                 Decision("r1-p3", "merge", "covered by Alpha", into="Alpha"),
                 Decision("r1-p4", "reject", "not cooperation")]
    schema, kept, log = apply_decisions(small_schema, props(), decisions, round_no=1)
    assert kept == ["Standard-setting"]
    cat = schema.category("Standard-setting")
    assert cat.origin == "discovered" and cat.source == "theory ref" and "Drafting rules." in cat.include
    assert "Sharing technology." in schema.category("Alpha").include
    assert {e["action"] for e in log} == {"keep", "merge", "reject"}
    assert [e["into"] for e in log if e["pid"] == "r1-p2"] == ["Standard-setting"]


@pytest.mark.parametrize("bad", [
    [Decision("r1-p1", "keep", "")],                               # no rationale
    [Decision("r1-p1", "maybe", "x")],                             # unknown action
    [Decision("r1-p1", "merge", "x", into="Nowhere")],             # bad merge target
])
def test_invalid_decisions(small_schema, bad):
    others = [Decision(p.pid, "reject", "r") for p in props()[1:]]
    with pytest.raises(ValueError):
        apply_decisions(small_schema, props(), bad + others, 1)


def test_missing_decision_rejected(small_schema):
    with pytest.raises(ValueError):
        apply_decisions(small_schema, props(), [Decision("r1-p1", "keep", "x")], 1)


def test_file_adjudicator_roundtrip(small_schema, tmp_path):
    adj = FileAdjudicator(tmp_path)
    with pytest.raises(AwaitingHumanInput):
        adj.decide(props(), small_schema, "round1")
    path = tmp_path / "round1" / "proposals.yaml"
    doc = yaml.safe_load(path.read_text())
    with pytest.raises(AwaitingHumanInput):        # not ready yet
        adj.decide(props(), small_schema, "round1")
    for item in doc["proposals"]:
        item["decision"]["action"] = "keep" if item["pid"] == "r1-p1" else "reject"
        item["decision"]["rationale"] = "because"
    doc["ready"] = True
    path.write_text(yaml.safe_dump(doc))
    decisions = adj.decide(props(), small_schema, "round1")
    assert [d.action for d in decisions] == ["keep", "reject", "reject", "reject"]
    assert adj.decide([], small_schema, "round2") == []
