import csv
import io

import pytest

from aasl.annotation import (FileAnnotator, label_to_row, parse_row, read_labels, write_labels)
from aasl.core import AwaitingHumanInput, Label, ReliabilityTooLow, TextUnit


def units(n):
    return [TextUnit(f"u{i}", f"text {i}") for i in range(n)]


def lab(cat, sub=None):
    return Label(True, cat, {"form": ["Contract"], "partners": ["P"]}, sub)


def test_files_saved_by_excel_are_read(tmp_path, sub_schema):
    us = units(3)
    labels = {u.uid: Label(True, "Alpha", {"form": ["Contract"], "partners": ["A", "B"]}, "S1")
              for u in us}
    p = tmp_path / "plain.csv"
    write_labels(p, us, labels, sub_schema)
    base = p.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(base)))
    variants = {"bom": ("utf-8-sig", base)}
    for name, delim in (("semicolon", ";"), ("tab", "\t")):
        buf = io.StringIO()
        csv.writer(buf, delimiter=delim).writerows(rows)
        variants[name] = ("utf-8-sig", buf.getvalue())
    for name, (enc, text) in variants.items():
        f = tmp_path / f"{name}.csv"
        f.write_text(text, encoding=enc)
        got, missing, problems = read_labels(f, sub_schema)
        assert not missing and not problems, name
        assert got["u1"].subcategory == "S1" and got["u1"].attributes["partners"] == ["A", "B"], name


def test_subcategory_parsing(sub_schema):
    base = {"uid": "x", "in_domain": "yes", "category": "Alpha", "attr:form": "Contract"}
    ok, probs = parse_row({**base, "subcategory": "S2"}, sub_schema)
    assert ok.subcategory == "S2" and not probs
    blank, probs = parse_row({**base, "subcategory": ""}, sub_schema)
    assert blank.subcategory is None and not probs
    bad, probs = parse_row({**base, "subcategory": "Nope"}, sub_schema)
    assert any("subcategory" in p for p in probs)
    beta, probs = parse_row({"uid": "y", "in_domain": "yes", "category": "Beta",
                             "subcategory": "S1"}, sub_schema)
    assert beta.subcategory is None            # Beta defines none; the value is ignored
    row = label_to_row(TextUnit("x", "t"), Label(True, "Alpha", {}, "S1"), sub_schema)
    assert row["subcategory"] == "S1"


def test_unknown_ids_and_uncoded_rows_are_reported(tmp_path, sub_schema):
    us = units(4)
    ann = FileAnnotator(tmp_path, 1)
    with pytest.raises(AwaitingHumanInput):
        ann.annotate(us, sub_schema, "t")
    labels = {u.uid: lab("Beta") for u in us}
    write_labels(tmp_path / "t" / "coder_1.csv", us, labels, sub_schema)
    text = (tmp_path / "t" / "coder_1.csv").read_text().replace("u2,", "2.0,")   # id mangled
    (tmp_path / "t" / "coder_1.csv").write_text(text)
    with pytest.raises(AwaitingHumanInput):
        ann.annotate(us, sub_schema, "t")
    problems = (tmp_path / "t" / "PROBLEMS.txt").read_text()
    assert "not in this sample" in problems and "1 units not coded yet" in problems


def _two_coders(tmp_path, schema, us, l1, l2, **kw):
    ann = FileAnnotator(tmp_path, 2, **kw)
    with pytest.raises(AwaitingHumanInput):
        ann.annotate(us, schema, "t")
    write_labels(tmp_path / "t" / "coder_1.csv", us, l1, schema)
    write_labels(tmp_path / "t" / "coder_2.csv", us, l2, schema)
    return ann


def test_low_reliability_stops_before_anyone_resolves_anything(tmp_path, sub_schema):
    us = units(40)
    cats = ["Alpha", "Beta"]
    l1 = {u.uid: lab(cats[i % 2], "S1" if i % 2 == 0 else None) for i, u in enumerate(us)}
    l2 = {u.uid: lab(cats[(i // 2) % 2], "S1" if (i // 2) % 2 == 0 else None)
          for i, u in enumerate(us)}
    ann = _two_coders(tmp_path, sub_schema, us, l1, l2, min_reliability=0.7)
    with pytest.raises(ReliabilityTooLow, match="No disagreements were sent"):
        ann.annotate(us, sub_schema, "t")
    assert (tmp_path / "t" / "RELIABILITY.txt").exists()
    assert not (tmp_path / "t" / "resolution.csv").exists()
    # with the gate set to warn, the run proceeds to the resolution step
    warn = FileAnnotator(tmp_path, 2, min_reliability=0.7, on_low_reliability="warn")
    with pytest.raises(AwaitingHumanInput, match="disagreements"):
        warn.annotate(us, sub_schema, "t")
    assert (tmp_path / "t" / "resolution.csv").exists()


def test_resolution_survives_coders_editing_their_files(tmp_path, sub_schema):
    us = units(8)
    base = {u.uid: lab("Alpha", "S1") for u in us}
    c2 = dict(base)
    c2["u1"], c2["u3"] = lab("Beta"), lab("Beta")
    ann = _two_coders(tmp_path, sub_schema, us, base, c2)
    with pytest.raises(AwaitingHumanInput):
        ann.annotate(us, sub_schema, "t")
    res = tmp_path / "t" / "resolution.csv"
    rows = list(csv.DictReader(open(res, encoding="utf-8")))
    assert [r["uid"] for r in rows] == ["u1", "u3"]
    for r in rows:
        r.update({"final:in_domain": "yes", "final:category": "Alpha", "final:subcategory": "S1",
                  "final:attr:form": "Contract", "final:attr:partners": "P", "rationale": "ok"})
    with open(res, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # coder 2 revises: u3 now agrees, u5 now disagrees
    c2b = dict(base)
    c2b["u1"], c2b["u5"] = lab("Beta"), lab("Beta")
    write_labels(tmp_path / "t" / "coder_2.csv", us, c2b, sub_schema)
    with pytest.raises(AwaitingHumanInput):                      # u5 is new and unresolved
        ann.annotate(us, sub_schema, "t")
    new = list(csv.DictReader(open(res, encoding="utf-8")))
    assert [r["uid"] for r in new] == ["u1", "u5"]
    assert new[0]["final:category"] == "Alpha" and new[1]["final:category"] == ""   # carried over
    assert (tmp_path / "t" / "resolution.stale.csv").exists()
    new[1].update({"final:in_domain": "yes", "final:category": "Beta"})
    with open(res, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new[0].keys()))
        w.writeheader()
        w.writerows(new)
    out = ann.annotate(us, sub_schema, "t")
    assert out.gold["u1"].category == "Alpha" and out.gold["u5"].category == "Beta"
    assert out.n_disagreements == 2
