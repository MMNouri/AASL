import json

from aasl.coder import Coder
from aasl.core import RESIDUAL, TextUnit
from aasl.llm.base import CachedModel, LanguageModel


class ScriptModel(LanguageModel):
    """Replies from a table: (task, uid) -> list of replies, consumed in order."""
    name = "script"

    def __init__(self, table):
        self.table = {k: list(v) for k, v in table.items()}
        self.calls = []

    def generate(self, system, user, schema, task):
        uid = user.split('<unit id="')[1].split('"')[0]
        self.calls.append((task, uid))
        replies = self.table[(task, uid)]
        return replies.pop(0) if len(replies) > 1 else replies[0]


U = [TextUnit("a", "Firm A and NGO B co-designed a pump."),
     TextUnit("b", "Firm A cut emissions."),
     TextUnit("c", "Firm A and City C did something new.")]


def test_three_step_coding_and_residual(small_schema, tmp_path):
    m = ScriptModel({
        ("detect", "a"): ['{"in_domain": true, "evidence": "co-designed a pump"}'],
        ("assign", "a"): ['{"category": "Alpha", "evidence": "co-designed a pump"}'],
        ("attributes", "a"): ['{"form": "NOT_STATED", "partners": ["NGO B"]}'],
        ("detect", "b"): ['{"in_domain": false, "evidence": ""}'],
        ("detect", "c"): ['{"in_domain": true, "evidence": "with City C"}'],
        ("assign", "c"): ['{"category": "RESIDUAL", "evidence": "did something new"}'],
    })
    codes = Coder(m, small_schema, max_workers=1).code_many(U)
    a, b, c = codes
    assert a.label.category == "Alpha" and a.label.attributes == {"form": [], "partners": ["NGO B"]}
    assert a.evidence_verified == {"detect": True, "assign": True}
    assert not b.label.in_domain and ("assign", "b") not in m.calls     # no step 2 when out of domain
    assert c.label.category == RESIDUAL and ("attributes", "c") not in m.calls
    assert c.evidence_verified["detect"] is False                         # quote not in text


def test_invalid_label_is_retried_then_recorded_as_error(small_schema):
    m = ScriptModel({
        ("detect", "a"): ['{"in_domain": true, "evidence": ""}'],
        ("assign", "a"): ['{"category": "Gamma", "evidence": ""}'],         # never valid
    })
    code = Coder(m, small_schema, retries=2, max_workers=1).code_unit(U[0])
    assert code.status == "error" and "Gamma" in code.error
    assert sum(1 for t, _ in m.calls if t == "assign") == 3


def test_retry_recovers(small_schema):
    m = ScriptModel({
        ("detect", "a"): ['not json', '{"in_domain": false, "evidence": ""}'],
    })
    code = Coder(m, small_schema, max_workers=1).code_unit(U[0])
    assert code.status == "ok" and not code.label.in_domain


def test_checkpoint_resume_and_schema_change(small_schema, tmp_path):
    table = {("detect", u.uid): ['{"in_domain": false, "evidence": ""}'] for u in U}
    m = ScriptModel(table)
    ck = tmp_path / "ck.jsonl"
    Coder(m, small_schema, max_workers=2).code_many(U, checkpoint=ck)
    n = len(m.calls)
    Coder(m, small_schema, max_workers=2).code_many(U, checkpoint=ck)
    assert len(m.calls) == n                                  # all reused
    changed = small_schema.with_notes({"domain": ["new note"]})
    Coder(m, changed, max_workers=2).code_many(U, checkpoint=ck)
    assert len(m.calls) == 2 * n                              # different schema: recoded
    assert all(json.loads(l)["uid"] for l in ck.read_text().splitlines())


def test_cache_makes_reruns_free(small_schema, tmp_path):
    table = {("detect", u.uid): ['{"in_domain": false, "evidence": ""}'] for u in U}
    inner = ScriptModel(table)
    cached = CachedModel(inner, tmp_path / "cache")
    Coder(cached, small_schema, max_workers=1).code_many(U)
    Coder(cached, small_schema, max_workers=1).code_many(U)
    assert len(inner.calls) == 3 and cached.hits == 3


def test_empty_citation_carries_no_flag_and_a_wrong_quote_is_flagged(small_schema):
    m = ScriptModel({
        ("detect", "a"): ['{"in_domain": false, "evidence": ""}'],           # nothing to cite
        ("detect", "b"): ['{"in_domain": false, "evidence": "text that is not in the unit"}'],
    })
    coder = Coder(m, small_schema, max_workers=1)
    a = coder.code_unit(U[0])
    b = coder.code_unit(U[1])
    assert a.evidence_verified == {} and a.evidence == {"detect": None}
    assert b.evidence_verified == {"detect": False}


def test_subcategory_step_only_for_categories_that_define_them(sub_schema):
    m = ScriptModel({
        ("detect", "a"): ['{"in_domain": true, "evidence": ""}'],
        ("assign", "a"): ['{"category": "Alpha", "evidence": ""}'],
        ("subcategory", "a"): ['{"subcategory": "S2"}'],
        ("attributes", "a"): ['{"form": "NOT_STATED", "partners": []}'],
        ("detect", "b"): ['{"in_domain": true, "evidence": ""}'],
        ("assign", "b"): ['{"category": "Beta", "evidence": ""}'],
        ("attributes", "b"): ['{"form": "NOT_STATED", "partners": []}'],
        ("detect", "c"): ['{"in_domain": true, "evidence": ""}'],
        ("assign", "c"): ['{"category": "Alpha", "evidence": ""}'],
        ("subcategory", "c"): ['{"subcategory": "NONE_FITS"}'],
        ("attributes", "c"): ['{"form": "NOT_STATED", "partners": []}'],
    })
    a, b, c = Coder(m, sub_schema, max_workers=1).code_many(U)
    assert a.label.subcategory == "S2"
    assert b.label.subcategory is None and ("subcategory", "b") not in m.calls
    assert c.label.subcategory is None                        # NONE_FITS is stored as no subcategory


def test_a_dead_model_stops_the_run_instead_of_recording_thousands_of_errors(small_schema):
    import pytest as _pytest

    from aasl.core import ModelUnavailable

    class Dead(LanguageModel):
        name = "dead"

        def generate(self, system, user, schema, task):
            raise RuntimeError("401 invalid api key")

    many = [TextUnit(f"u{i}", "text") for i in range(30)]
    with _pytest.raises(ModelUnavailable, match="401 invalid api key"):
        Coder(Dead(), small_schema, retries=0, max_workers=2).code_many(many)
