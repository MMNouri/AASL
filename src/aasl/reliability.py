"""Inter-coder reliability for the human-coded samples.

The human labels are the yardstick for every precision and recall figure, so
their reliability is reported with every sample. Cohen's kappa is given for
two coders; Krippendorff's alpha (nominal) handles any number of coders and
missing codes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations
from typing import Optional


def percent_agreement(a: list, b: list) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def cohen_kappa(a: list, b: list) -> Optional[float]:
    """Cohen's kappa for two coders over the units both coded."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def krippendorff_alpha_nominal(coders: list) -> Optional[float]:
    """Krippendorff's alpha, nominal metric.

    ``coders`` is a list of equal-length lists (one per coder); ``None`` marks a
    missing code. Units with fewer than two codes are not pairable and drop out.
    """
    if not coders:
        return None
    n_units = len(coders[0])
    if any(len(c) != n_units for c in coders):
        raise ValueError("all coders must code the same units")
    o = defaultdict(float)
    for u in range(n_units):
        values = [c[u] for c in coders if c[u] is not None]
        m = len(values)
        if m < 2:
            continue
        for i, j in permutations(range(m), 2):
            o[(values[i], values[j])] += 1.0 / (m - 1)
    n_c = defaultdict(float)
    for (c, _k), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1:
        return None
    observed = sum(v for (c, k), v in o.items() if c != k)
    expected = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k)
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - (n - 1) * observed / expected


def reliability_report(coder_labels: list) -> dict:
    """Summary for a list of per-coder label lists (flat labels)."""
    rep = {"n_coders": len(coder_labels), "n_units": len(coder_labels[0]) if coder_labels else 0,
           "alpha_nominal": krippendorff_alpha_nominal(coder_labels)}
    if len(coder_labels) == 2:
        rep["cohen_kappa"] = cohen_kappa(*coder_labels)
        rep["percent_agreement"] = percent_agreement(*coder_labels)
    return rep
