"""Persisted run state.

A run pauses whenever it needs a researcher (coding, prompt revision,
adjudication). Everything decided so far is saved here, so re-running picks
up at the same point and the run directory doubles as the audit record.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path


class RunStore:
    def __init__(self, run_dir):
        self.root = Path(run_dir)
        self.state = self.root / "state"
        self.state.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.state / f"{key.replace('/', '__')}.json"

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str):
        return json.loads(self._path(key).read_text(encoding="utf-8"))

    def save(self, key: str, obj) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(path)

    def log(self, event: str, **details) -> None:
        rec = {"time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "event": event, **details}
        with open(self.root / "events.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def path(self, *parts) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
