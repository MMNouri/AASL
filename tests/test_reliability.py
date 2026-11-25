import pytest

from aasl.reliability import cohen_kappa, krippendorff_alpha_nominal, percent_agreement


def test_cohen_kappa_known_value():
    a = ["x", "x", "y", "y", "x", "y", "x", "x", "y", "x"]
    b = ["x", "y", "y", "y", "x", "y", "x", "x", "x", "x"]
    # po = 0.8; pe = (0.6*0.6 + 0.4*0.4) = 0.52; kappa = 0.28/0.48
    assert cohen_kappa(a, b) == pytest.approx(0.28 / 0.48)
    assert percent_agreement(a, b) == pytest.approx(0.8)


def test_kappa_perfect_and_constant():
    assert cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_krippendorff_reference_example():
    # Krippendorff's reliability-data example (4 coders, 12 units, missing data);
    # nominal alpha = 0.743 (cross-checked against the `krippendorff` package).
    n = None
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
        [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
    ]
    assert krippendorff_alpha_nominal(data) == pytest.approx(0.743, abs=1e-3)


def test_alpha_perfect_agreement():
    assert krippendorff_alpha_nominal([["a", "b", "a"], ["a", "b", "a"]]) == 1.0
