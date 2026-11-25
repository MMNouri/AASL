"""Build a run from a YAML config file (see examples/cooperation/config.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

from .adjudication import FileAdjudicator
from .annotation import FileAnnotator
from .calibration import FileReviser
from .io import load_units
from .llm.factory import build_model
from .pipeline import AASLRun, Settings
from .schema import Schema


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg["_base"] = str(path.parent.resolve())
    for key in ("run_dir", "data", "schema", "model"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    return cfg


def _resolve(base: str, p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(base) / p


def build_run(cfg: dict, verbose: bool = True) -> AASLRun:
    base = cfg["_base"]
    run_dir = _resolve(base, cfg["run_dir"])
    data = cfg["data"]
    units = load_units(_resolve(base, data["path"]), data.get("id_column", "id"),
                       data.get("text_column", "text"), data.get("meta_columns", []))
    schema = Schema.load(_resolve(base, cfg["schema"]))
    model = build_model(cfg["model"], cache_dir=run_dir / "llm_cache")
    settings = Settings.from_dict(cfg.get("settings", {}))
    human = cfg.get("human", {})
    annotator = FileAnnotator(run_dir / "human" / "coding", human.get("n_coders", 2),
                              min_reliability=settings.min_reliability,
                              reliability_metric=settings.reliability_metric,
                              on_low_reliability=settings.on_low_reliability)
    return AASLRun(units, schema, model, annotator=annotator,
                   reviser=FileReviser(run_dir / "human" / "revisions"),
                   adjudicator=FileAdjudicator(run_dir / "human" / "adjudication"),
                   settings=settings, run_dir=run_dir, verbose=verbose)
