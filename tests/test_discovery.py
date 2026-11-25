import json

from aasl.core import RESIDUAL, Coding, Label, TextUnit
from aasl.discovery import collect_residual, propose_categories
from aasl.llm.base import LanguageModel

UNITS = {
    "r1": TextUnit("r1", "We co-wrote the sector code of conduct with peers."),
    "r2": TextUnit("r2", "Members drafted a shared certification standard."),
    "r3": TextUnit("r3", "We gave grants through a foundation."),
    "a1": TextUnit("a1", "Joint workshop with a university."),
    "o1": TextUnit("o1", "We cut emissions."),
}


def codings():
    return [Coding("r1", Label(True, RESIDUAL)), Coding("r2", Label(True, RESIDUAL)),
            Coding("r3", Label(True, RESIDUAL)), Coding("a1", Label(True, "Alpha")),
            Coding("o1", Label(False)), Coding("e1", None, status="error")]


def test_collect_residual_excludes_out_of_domain_and_errors():
    resid, n_dom, cov = collect_residual(codings())
    assert resid == ["r1", "r2", "r3"] and n_dom == 4 and cov == 0.25


class Proposer(LanguageModel):
    name = "proposer"

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, system, user, schema, task):
        self.prompts.append(user)
        return json.dumps(self.reply)


def test_only_residual_units_are_shown_and_quotes_are_verified(small_schema):
    reply = {"proposals": [
        {"name": "Standard setting", "definition": "Writing shared rules.",
         "supporting": [{"unit_id": "r1", "quote": "co-wrote the sector code of conduct"},
                        {"unit_id": "r2", "quote": "drafted a shared certification standard"}]},
        {"name": "Invented", "definition": "Made up.",
         "supporting": [{"unit_id": "r1", "quote": "text that is not there"},
                        {"unit_id": "a1", "quote": "Joint workshop"}]},     # a1 is not residual
        {"name": "alpha", "definition": "dup", "supporting": [
            {"unit_id": "r1", "quote": "We co-wrote"}, {"unit_id": "r2", "quote": "Members"}]},
    ]}
    m = Proposer(reply)
    res = propose_categories(m, small_schema, codings(), UNITS, min_units=2, round_no=1)
    shown = m.prompts[0]
    assert 'id="r1"' in shown and 'id="a1"' not in shown and 'id="o1"' not in shown
    assert [p.name for p in res.proposals] == ["Standard setting"]
    assert res.proposals[0].unit_ids == ["r1", "r2"]
    reasons = {d["name"]: d["reason"] for d in res.dropped}
    assert "not found" in reasons["Invented"]
    assert "duplicates an existing category" in reasons["alpha"]


def test_same_proposal_across_batches_is_merged(small_schema):
    reply = {"proposals": [{"name": "Standard Setting", "definition": "x", "supporting": [
        {"unit_id": "r1", "quote": "We co-wrote"}, {"unit_id": "r2", "quote": "Members drafted"},
        {"unit_id": "r3", "quote": "gave grants"}]}]}
    res = propose_categories(Proposer(reply), small_schema, codings(), UNITS, batch_size=2,
                             min_units=2)
    # batch 1 = r1,r2 ; batch 2 = r3 alone (< min_units, not sent)
    assert len(res.proposals) == 1 and res.proposals[0].unit_ids == ["r1", "r2"]


def test_a_repeat_of_a_rejected_proposal_is_dropped_even_if_the_model_ignores_the_instruction(small_schema):
    reply = {"proposals": [{"name": "Standard Setting", "definition": "x", "supporting": [
        {"unit_id": "r1", "quote": "We co-wrote"}, {"unit_id": "r2", "quote": "Members drafted"}]}]}
    m = Proposer(reply)
    rejected = [{"name": "standard setting", "definition": "Writing rules.",
                 "rationale": "not cooperation"}]
    res = propose_categories(m, small_schema, codings(), UNITS, rejected=rejected)
    assert res.proposals == []
    assert [d["reason"] for d in res.dropped] == ["already rejected by the researcher in an earlier round"]
    assert "not cooperation" in m.prompts[0]                  # the model was shown why
