"""Load text units U from CSV, JSONL or Parquet."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .core import TextUnit


def load_units(path, id_column: str = "id", text_column: str = "text",
               meta_columns: list = None, drop_empty: bool = True) -> list:
    """Read units from a file. ``meta_columns`` are kept for stratification and
    clustering (e.g. firm, sector, region, year)."""
    path = Path(path)
    meta_columns = list(meta_columns or [])
    suffix = path.suffix.lower()
    if suffix == ".csv":
        from .annotation import read_csv_rows
        rows = read_csv_rows(path)
    elif suffix in (".jsonl", ".ndjson"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("reading parquet needs pandas and pyarrow") from exc
        rows = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise ValueError(f"unsupported file type {suffix!r} (csv, jsonl, parquet)")
    units, seen = [], set()
    for i, row in enumerate(rows):
        missing = [c for c in [id_column, text_column] if c not in row]
        if missing:
            raise ValueError(f"row {i}: missing columns {missing}")
        uid = str(row[id_column])
        text = "" if row[text_column] is None else str(row[text_column])
        if drop_empty and not text.strip():
            continue
        if uid in seen:
            raise ValueError(f"duplicate id {uid!r}")
        seen.add(uid)
        meta = {c: ("" if row.get(c) is None else str(row.get(c))) for c in meta_columns}
        units.append(TextUnit(uid, text, meta))
    return units


def write_units_csv(path, units: list, meta_columns: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text"] + list(meta_columns))
        for u in units:
            w.writerow([u.uid, u.text] + [u.meta.get(c, "") for c in meta_columns])
