import pytest
import yaml

from aasl.calibration import CallbackReviser, FileReviser, calibrate
from aasl.core import AwaitingHumanInput, CalibrationFailed, Coding, Label, TextUnit

UNITS = [TextUnit(f"u{i}", f"text {i}") for i in range(20)]
GOLD = {u.uid: Label(True, "Alpha" if i < 10 else "Beta") for i, u in enumerate(UNITS)}


def coder_that_needs_a_note(schema, units, name):
    fixed = bool(schema.notes_for("Alpha"))
    out = []
    for i, u in enumerate(units):
        cat = GOLD[u.uid].category
        if not fixed and i < 4:
            cat = "Beta"                     # confuses Alpha until clarified
        out.append(Coding(u.uid, Label(True, cat)))
    return out


def test_loop_revises_until_threshold(small_schema):
    calls = []

    def fix(schema, report, errors, tag):
        calls.append((tag, len(errors)))
        return schema.with_notes({"Alpha": ["clarified"]})

    res = calibrate(small_schema, UNITS, GOLD, coder_that_needs_a_note, CallbackReviser(fix),
                    tau=0.9, min_support=5, max_iterations=4, tag="r1")
    assert res.report.passes and len(res.iterations) == 2
    assert calls == [("r1_iter1", 4)]
    assert res.schema.notes_for("Alpha") == ["clarified"]


def test_revision_may_not_change_categories(small_schema):
    from aasl.schema import Category

    def cheat(schema, report, errors, tag):
        return schema.with_category(Category("Gamma", "new"))

    with pytest.raises(ValueError, match="category set"):
        calibrate(small_schema, UNITS, GOLD, coder_that_needs_a_note, CallbackReviser(cheat),
                  tau=0.9, min_support=5, max_iterations=3, tag="r1")


def test_gives_up_after_max_iterations(small_schema):
    def noop(schema, report, errors, tag):
        return schema.with_notes({"domain": [tag]})

    def never_right(schema, units, name):
        return [Coding(u.uid, Label(True, "Beta")) for u in units]

    with pytest.raises(CalibrationFailed):
        calibrate(small_schema, UNITS, GOLD, never_right, CallbackReviser(noop),
                  tau=0.9, min_support=5, max_iterations=3, tag="r1")


def test_file_reviser(small_schema, tmp_path):
    rev = FileReviser(tmp_path)

    def run():
        return calibrate(small_schema, UNITS, GOLD, coder_that_needs_a_note, rev, tau=0.9,
                         min_support=5, max_iterations=3, tag="r1")

    with pytest.raises(AwaitingHumanInput):
        run()
    folder = tmp_path / "r1_iter1"
    assert (folder / "report.md").exists() and (folder / "disagreements.csv").exists()
    doc = yaml.safe_load((folder / "revision.yaml").read_text())
    doc["notes"]["Alpha"] = ["Alpha covers the first ten units."]
    doc["ready"] = True
    (folder / "revision.yaml").write_text(yaml.safe_dump(doc))
    res = run()
    assert res.report.passes and res.schema.notes_for("Alpha")
