"""The model coder M: codes units into the domain, a category or the residual,
and extracts subcategory and attributes.

Every call is validated against its schema. A unit whose calls still fail
after retries gets ``status="error"``: it is counted against the model in
evaluation and listed in the error report, never silently filled. If most
calls in a batch fail (bad key, outage) the run stops instead of recording
thousands of errors.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from .core import RESIDUAL, Coding, Label, ModelUnavailable, TextUnit, quote_in_text
from .llm.base import LanguageModel, call_json
from .prompts import (NOT_STATED, assign_prompt, attributes_prompt, detect_prompt,
                      subcategory_prompt)
from .schema import Schema

FAIL_FAST_RATIO = 0.5        # abort when more than this share of a batch errors...
FAIL_FAST_MIN_BATCH = 10     # ...provided the batch has at least this many units


class Coder:
    def __init__(self, model: LanguageModel, schema: Schema, retries: int = 2,
                 max_workers: int = 8, backoff: float = 0.0):
        self.model = model
        self.schema = schema
        self.retries = retries
        self.max_workers = max_workers
        self.backoff = backoff
        self.fingerprint = schema.fingerprint()

    # -------------------------------------------------------------- one unit
    def _ask(self, prompt) -> dict:
        return call_json(self.model, prompt.system, prompt.user, prompt.schema, prompt.task,
                         retries=self.retries, backoff=self.backoff)

    @staticmethod
    def _cite(evidence: dict, verified: dict, key: str, quote: Optional[str], text: str) -> None:
        """Record a cited passage and whether it occurs in the unit. An empty
        citation is not evidence, so it carries no verification flag."""
        quote = (quote or "").strip()
        evidence[key] = quote or None
        if quote:
            verified[key] = quote_in_text(quote, text)

    def code_unit(self, unit: TextUnit) -> Coding:
        evidence, verified = {}, {}
        try:
            det = self._ask(detect_prompt(self.schema, unit))
            self._cite(evidence, verified, "detect", det.get("evidence"), unit.text)
            if not det["in_domain"]:
                label = Label(in_domain=False)
            else:
                asg = self._ask(assign_prompt(self.schema, unit))
                category = asg["category"]
                self._cite(evidence, verified, "assign", asg.get("evidence"), unit.text)
                sub, attrs = None, {}
                if category != RESIDUAL:
                    if self.schema.category(category).subcategories:
                        sub = self._ask(subcategory_prompt(self.schema, unit, category))["subcategory"]
                    if self.schema.attributes_for(category):
                        raw = self._ask(attributes_prompt(self.schema, unit, category))
                        attrs = _clean_attributes(raw)
                label = Label(in_domain=True, category=category, attributes=attrs, subcategory=sub)
        except Exception as exc:  # recorded, not hidden
            return Coding(uid=unit.uid, label=None, status="error", error=str(exc)[:500],
                          evidence=evidence, evidence_verified=verified,
                          schema_fingerprint=self.fingerprint, model_name=self.model.name)
        return Coding(uid=unit.uid, label=label, evidence=evidence, evidence_verified=verified,
                      schema_fingerprint=self.fingerprint, model_name=self.model.name)

    # ------------------------------------------------------------ many units
    def code_many(self, units: Iterable[TextUnit], checkpoint: Optional[Path] = None,
                  progress: bool = False) -> list:
        """Code ``units``; with ``checkpoint`` each finished coding is appended to
        a JSONL file so an interrupted run resumes where it stopped. Codings from
        a different schema version are never reused. Work is submitted in bounded
        chunks so memory stays flat on large corpora."""
        units = list(units)
        done = _load_checkpoint(checkpoint, self.fingerprint) if checkpoint else {}
        todo = [u for u in units if u.uid not in done]
        lock = threading.Lock()
        fh = None
        if checkpoint and todo:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            fh = open(checkpoint, "a", encoding="utf-8")
        chunk = max(FAIL_FAST_MIN_BATCH, 50 * max(1, self.max_workers))
        finished = 0
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
                for start in range(0, len(todo), chunk):
                    batch = todo[start:start + chunk]
                    futures = [pool.submit(self.code_unit, u) for u in batch]
                    errors = []
                    for fut in as_completed(futures):
                        coding = fut.result()
                        with lock:
                            done[coding.uid] = coding
                            if fh:
                                fh.write(json.dumps(coding.to_dict(), ensure_ascii=False) + "\n")
                                fh.flush()
                        if coding.status == "error":
                            errors.append(coding.error)
                    finished += len(batch)
                    if progress and finished % 500 < len(batch):
                        print(f"  coded {finished}/{len(todo)}")
                    if (len(batch) >= FAIL_FAST_MIN_BATCH
                            and len(errors) > FAIL_FAST_RATIO * len(batch)):
                        raise ModelUnavailable(
                            f"{len(errors)} of {len(batch)} units in a batch failed; the model "
                            f"service looks unavailable or misconfigured. First error: {errors[0]}")
        finally:
            if fh:
                fh.close()
        return [done[u.uid] for u in units]


def _clean_attributes(raw: dict) -> dict:
    out = {}
    for name, value in raw.items():
        values = value if isinstance(value, list) else [value]
        kept = [str(v).strip() for v in values if v is not None and str(v).strip()
                and str(v).strip() != NOT_STATED]
        out[name] = list(dict.fromkeys(kept))
    return out


def _load_checkpoint(path: Optional[Path], fingerprint: str) -> dict:
    if not path or not Path(path).exists():
        return {}
    done = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            c = Coding.from_dict(json.loads(line))
        except (json.JSONDecodeError, KeyError):
            continue  # a line cut off by an interruption
        # errored units are retried when the run resumes
        if c.schema_fingerprint == fingerprint and c.status == "ok":
            done[c.uid] = c
    return done
