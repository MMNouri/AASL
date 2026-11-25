"""Stratified sampling and the sample ledger.

The ledger records every unit drawn and why. It enforces the separation the
algorithm depends on:

* the held-out sample is drawn first and never enters calibration or
  discovery, so the final check (line 13) is untouched by schema building;
* each round's calibration sample A is fresh, so a schema changed by
  discovery is calibrated on units it was not built from.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .core import TextUnit


def _stratum(unit: TextUnit, keys: list) -> tuple:
    return tuple(str(unit.meta.get(k, "NA")) for k in keys)


def stratified_sample(units: list, n: int, strata: Optional[list] = None,
                      cluster: Optional[str] = None, max_per_cluster: Optional[int] = None,
                      exclude: Optional[set] = None, seed: int = 0,
                      cluster_counts: Optional[dict] = None) -> list:
    """Draw ``n`` units with proportional allocation across strata.

    Allocation uses the largest-remainder method. ``max_per_cluster`` caps how
    many units one cluster (e.g. one firm) can contribute, so a few verbose
    clusters cannot dominate the sample. Deterministic for a given seed.
    """
    strata = strata or []
    exclude = exclude or set()
    rng = random.Random(seed)
    pool = [u for u in units if u.uid not in exclude]
    if n >= len(pool):
        return sorted(pool, key=lambda u: u.uid)

    groups = defaultdict(list)
    for u in sorted(pool, key=lambda u: u.uid):
        groups[_stratum(u, strata)].append(u)
    keys = sorted(groups)
    total = len(pool)
    quotas = {k: n * len(groups[k]) / total for k in keys}
    alloc = {k: int(quotas[k]) for k in keys}
    remainder = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), str(k)))[:remainder]:
        alloc[k] += 1

    per_cluster = defaultdict(int, cluster_counts or {})
    chosen, leftovers = [], []
    for k in keys:
        members = groups[k][:]
        rng.shuffle(members)
        taken = 0
        for u in members:
            if taken >= alloc[k]:
                leftovers.append(u)
                continue
            c = u.meta.get(cluster) if cluster else None
            if max_per_cluster and c is not None and per_cluster[c] >= max_per_cluster:
                leftovers.append(u)
                continue
            chosen.append(u)
            taken += 1
            if c is not None:
                per_cluster[c] += 1
    # Fill any shortfall caused by the cluster cap from the remaining units.
    rng.shuffle(leftovers)
    for u in leftovers:
        if len(chosen) >= n:
            break
        c = u.meta.get(cluster) if cluster else None
        if max_per_cluster and c is not None and per_cluster[c] >= max_per_cluster:
            continue
        chosen.append(u)
        if c is not None:
            per_cluster[c] += 1
    return sorted(chosen, key=lambda u: u.uid)


class SampleLedger:
    """Which units were used for what. Persisted as JSON in the run directory."""

    HOLDOUT = "holdout"

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.samples: dict = {}
        if self.path and self.path.exists():
            self.samples = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.samples, indent=1), encoding="utf-8")

    def has(self, name: str) -> bool:
        return name in self.samples

    def get(self, name: str) -> list:
        return list(self.samples[name])

    def used(self, exclude_names: tuple = ()) -> set:
        return {u for k, v in self.samples.items() if k not in exclude_names for u in v}

    def holdout_uids(self) -> set:
        """Every unit reserved as held-out (the base sample and any extensions)."""
        return {u for k, v in self.samples.items() if self._is_holdout(k) for u in v}

    def _is_holdout(self, name: str) -> bool:
        return name == self.HOLDOUT or name.startswith(self.HOLDOUT + "+")

    def record(self, name: str, uids: list) -> None:
        if name in self.samples:
            raise ValueError(f"sample {name!r} already recorded")
        uids = list(uids)
        if not self._is_holdout(name):
            leak = set(uids) & self.holdout_uids()
            if leak:
                raise ValueError(f"{len(leak)} held-out units would leak into {name!r}")
        self.samples[name] = uids
        self._save()

    def draw(self, name: str, units: list, n: int, *, fresh: bool = True, **kwargs) -> list:
        """Draw (or reload) a named sample. ``fresh`` excludes every unit already
        used; otherwise only the held-out units are excluded."""
        by_uid = {u.uid: u for u in units}
        if self.has(name):
            return [by_uid[u] for u in self.get(name)]
        exclude = self.used() if fresh else self.holdout_uids()
        sample = stratified_sample(units, n, exclude=exclude, **kwargs)
        self.record(name, [u.uid for u in sample])
        return sample

    def parts(self, name: str) -> list:
        """Names of the recorded parts of a sample: the base and its extensions."""
        out = [name] if self.has(name) else []
        k = 1
        while self.has(f"{name}+{k}"):
            out.append(f"{name}+{k}")
            k += 1
        return out

    def draw_extended(self, name: str, units: list, n: int, **kwargs) -> list:
        """Return a list of (part name, units) totalling ``n`` units. If the sample
        already exists with fewer units, a further random part is drawn from units
        not used anywhere else; the existing parts (and their human coding) are kept."""
        by_uid = {u.uid: u for u in units}
        existing = self.parts(name)
        if not existing:
            return [(name, self.draw(name, units, n, fresh=True, **kwargs))]
        have = sum(len(self.samples[p]) for p in existing)
        if n > have:
            cluster = kwargs.get("cluster")
            counts = defaultdict(int)
            if cluster:
                for p in existing:
                    for uid in self.samples[p]:
                        counts[by_uid[uid].meta.get(cluster)] += 1
            part = f"{name}+{len(existing)}"
            self.draw(part, units, n - have, fresh=True, cluster_counts=counts, **kwargs)
            existing = self.parts(name)
        return [(p, [by_uid[u] for u in self.samples[p]]) for p in existing]
