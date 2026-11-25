from collections import Counter

import pytest

from aasl.core import TextUnit
from aasl.sampling import SampleLedger, stratified_sample


def corpus(n=1000):
    return [TextUnit(f"u{i}", "t", {"sector": "A" if i % 4 else "B", "firm": f"f{i % 50}"})
            for i in range(n)]


def test_proportional_and_deterministic():
    units = corpus()
    s1 = stratified_sample(units, 200, strata=["sector"], seed=3)
    s2 = stratified_sample(units, 200, strata=["sector"], seed=3)
    assert [u.uid for u in s1] == [u.uid for u in s2]
    counts = Counter(u.meta["sector"] for u in s1)
    assert counts == {"A": 150, "B": 50}


def test_cluster_cap():
    units = corpus()
    s = stratified_sample(units, 100, cluster="firm", max_per_cluster=2, seed=1)
    assert len(s) == 100
    assert max(Counter(u.meta["firm"] for u in s).values()) <= 2


def test_ledger_keeps_holdout_out_and_samples_fresh(tmp_path):
    units = corpus()
    led = SampleLedger(tmp_path / "samples.json")
    hold = led.draw("holdout", units, 100, seed=1)
    a1 = led.draw("round1/calibration", units, 100, seed=2)
    a2 = led.draw("round2/calibration", units, 100, seed=3)
    ids = [{u.uid for u in s} for s in (hold, a1, a2)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    # reload from disk returns the same samples
    again = SampleLedger(tmp_path / "samples.json").draw("round1/calibration", units, 100, seed=99)
    assert {u.uid for u in again} == ids[1]
    with pytest.raises(ValueError):
        led.record("leak", [next(iter(ids[0]))])
