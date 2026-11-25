import pytest

from aasl.core import RESIDUAL, Coding, Label
from aasl.metrics import evaluate, required_sample_size


def L(cat=None, **attrs):
    if cat is None:
        return Label(False)
    return Label(True, cat, {k: v for k, v in attrs.items()})


def C(uid, label, status="ok"):
    return Coding(uid=uid, label=label, status=status)


def test_per_category_scores_hand_computed(small_schema):
    gold = {"1": L("Alpha"), "2": L("Alpha"), "3": L("Beta"), "4": L(), "5": L(RESIDUAL),
            "6": L("Alpha")}
    codes = [C("1", L("Alpha")), C("2", L("Beta")), C("3", L("Beta")), C("4", L("Alpha")),
             C("5", L(RESIDUAL)), C("6", None, status="error")]
    rep = evaluate(gold, codes, small_schema, tau=0.9, min_support=1)
    a = rep.categories["Alpha"]
    assert (a.support, a.predicted, a.tp) == (3, 2, 1)
    assert a.precision == pytest.approx(0.5) and a.recall == pytest.approx(1 / 3)
    b = rep.categories["Beta"]
    assert (b.precision, b.recall) == (0.5, 1.0)
    # detection: gold in-domain {1,2,3,5,6}; predicted in-domain {1,2,3,4,5}; error is a miss
    assert rep.detection.tp == 4 and rep.detection.predicted == 5 and rep.detection.support == 5
    assert rep.n_errors == 1 and not rep.passes
    assert rep.residual.precision == 1.0


def test_gate_requires_every_category_and_support(small_schema):
    gold = {str(i): L("Alpha") for i in range(20)}
    gold.update({f"b{i}": L("Beta") for i in range(3)})
    codes = [C(u, gold[u]) for u in gold]
    rep = evaluate(gold, codes, small_schema, tau=0.9, min_support=5)
    assert not rep.passes
    assert any("Beta" in f and "human-coded" in f for f in rep.failures)
    rep2 = evaluate(gold, codes, small_schema, tau=0.9, min_support=3)
    assert rep2.passes


def test_pooled_score_cannot_hide_a_failing_category(small_schema):
    gold = {f"a{i}": L("Alpha") for i in range(90)}
    gold.update({f"b{i}": L("Beta") for i in range(10)})
    codes = [C(u, gold[u]) for u in gold if u.startswith("a")]
    codes += [C(f"b{i}", L("Alpha") if i < 3 else L("Beta")) for i in range(10)]
    rep = evaluate(gold, codes, small_schema, tau=0.9, min_support=5)
    # 97% of units are right, but Beta recall is 0.7
    assert not rep.passes and any("Beta: recall" in f for f in rep.failures)


def test_attribute_scores_and_gating(small_schema):
    gold = {str(i): L("Alpha", form=["Contract"], partners=["X Corp"]) for i in range(12)}
    codes = [C(str(i), L("Alpha", form=["Contract"] if i < 6 else ["Equity"], partners=["x corp"]))
             for i in range(12)]
    gold.update({f"b{i}": L("Beta") for i in range(6)})     # every category needs support
    codes += [C(f"b{i}", L("Beta")) for i in range(6)]
    rep = evaluate(gold, codes, small_schema, tau=0.9, min_support=5)
    assert rep.attributes["form"].precision == pytest.approx(0.5)
    assert rep.attributes["partners"].recall == 1.0          # case-insensitive match
    assert rep.passes                                        # attributes not gated by default
    rep2 = evaluate(gold, codes, small_schema, tau=0.9, min_support=5, gate_attributes=True)
    assert not rep2.passes


def test_coverage_and_bootstrap(small_schema):
    gold = {f"u{i}": (L("Alpha") if i % 2 else L(RESIDUAL)) for i in range(40)}
    codes = [C(u, g) for u, g in gold.items()]
    clusters = {u: f"firm{int(u[1:]) % 8}" for u in gold}
    rep = evaluate(gold, codes, small_schema, tau=0.9, min_support=1, cluster_of=clusters,
                   n_boot=200, seed=1)
    assert rep.gold_coverage == pytest.approx(0.5) and rep.model_coverage == pytest.approx(0.5)
    assert rep.categories["Alpha"].ci_recall == [1.0, 1.0]
    assert "Result" in rep.to_markdown()


def test_required_sample_size():
    assert required_sample_size(0.9, 0.05) == 139
    assert required_sample_size(0.9, 0.05, design_effect=1.5) == 208


def test_subcategory_scores_use_only_units_where_the_category_agrees(sub_schema):
    gold = {f"g{i}": Label(True, "Alpha", {}, "S1" if i < 6 else "S2") for i in range(10)}
    gold["x"] = Label(True, "Alpha", {}, "S1")               # model will put this in Beta
    gold["b"] = Label(True, "Beta")
    preds = {f"g{i}": Label(True, "Alpha", {}, "S1" if i < 5 else "S2") for i in range(10)}
    preds["x"] = Label(True, "Beta")
    preds["b"] = Label(True, "Beta")
    codes = [Coding(u, l) for u, l in preds.items()]
    rep = evaluate(gold, codes, sub_schema, tau=0.5, min_support=1)
    s1, s2 = rep.subcategories["Alpha / S1"], rep.subcategories["Alpha / S2"]
    assert (s1.support, s1.predicted, s1.tp) == (6, 5, 5)           # x is excluded
    assert s1.precision == 1.0 and s1.recall == pytest.approx(5 / 6)
    assert (s2.support, s2.predicted, s2.tp) == (4, 5, 4)
    assert s2.precision == pytest.approx(0.8) and s2.recall == 1.0
    assert "Alpha / S1" in rep.to_markdown()
    strict = evaluate(gold, codes, sub_schema, tau=0.9, min_support=1)
    assert any("subcategory" in w for w in strict.warnings)          # reported ...
    assert not any("subcategory" in f for f in strict.failures)      # ... but not gated by default
    gated = evaluate(gold, codes, sub_schema, tau=0.9, min_support=1, gate_subcategories=True)
    assert any("subcategory Alpha / S2" in f for f in gated.failures)


def test_support_shortfalls_depend_on_human_labels_alone(small_schema):
    from aasl.metrics import support_shortfalls
    gold = {f"a{i}": Label(True, "Alpha") for i in range(90)}
    gold.update({f"b{i}": Label(True, "Beta") for i in range(10)})
    short = support_shortfalls(gold, small_schema, min_support=20)
    assert [(s["label"], s["support"], s["needed"]) for s in short] == [("Beta", 10, 240)]
    assert support_shortfalls(gold, small_schema, min_support=10) == []
    none_seen = {f"a{i}": Label(True, "Alpha") for i in range(50)}
    only = support_shortfalls(none_seen, small_schema, min_support=5)
    assert only == [{"label": "Beta", "support": 0, "n_units": 50, "needed": None}]
