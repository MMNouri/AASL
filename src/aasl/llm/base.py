"""The model interface used by every AASL stage, plus caching and retries."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..parsing import OutputError, parse_reply


class LanguageModel(ABC):
    """A model M. ``generate`` returns the raw text of one reply.

    ``schema`` is the JSON schema the reply must satisfy; adapters that support
    constrained decoding pass it to the provider, others rely on the prompt.
    ``task`` names the stage (detect, assign, attributes, propose) so adapters
    and logs can tell calls apart.
    """

    name: str = "model"

    @abstractmethod
    def generate(self, system: str, user: str, schema: dict, task: str) -> str:
        ...


class CachedModel(LanguageModel):
    """Disk cache keyed on (model, prompt, schema).

    Makes re-runs reproducible and free: when a run resumes after a human step,
    every earlier call is answered from the cache instead of the provider.
    """

    def __init__(self, inner: LanguageModel, cache_dir):
        self.inner = inner
        self.name = inner.name
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _key(self, system: str, user: str, schema: dict, task: str) -> str:
        blob = json.dumps([self.inner.name, task, system, user, schema], sort_keys=True,
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, system: str, user: str, schema: dict, task: str) -> str:
        key = self._key(system, user, schema, task)
        path = self.dir / key[:2] / f"{key}.json"
        if path.exists():
            with self._lock:
                self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["reply"]
        reply = self.inner.generate(system, user, schema, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"task": task, "reply": reply}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        with self._lock:
            self.misses += 1
        return reply

    def invalidate(self, system: str, user: str, schema: dict, task: str) -> None:
        key = self._key(system, user, schema, task)
        path = self.dir / key[:2] / f"{key}.json"
        if path.exists():
            path.unlink()


def call_json(model: LanguageModel, system: str, user: str, schema: dict, task: str,
              retries: int = 2, backoff: float = 0.0) -> dict:
    """Call the model and return a reply that validates against ``schema``.

    A reply that fails validation is retried with the problem stated in the
    prompt. Provider errors (network, rate limits) are retried with backoff.
    After ``retries`` extra attempts the last error is raised; the caller
    records it as a coding error.
    """
    last: Optional[Exception] = None
    prompt = user
    for attempt in range(retries + 1):
        try:
            raw = model.generate(system, prompt, schema, task)
        except Exception as exc:  # provider failure
            last = exc
            if backoff:
                time.sleep(backoff * (2 ** attempt))
            continue
        try:
            return parse_reply(raw, schema)
        except OutputError as exc:
            last = exc
            if isinstance(model, CachedModel):
                model.invalidate(system, prompt, schema, task)
            prompt = (user + "\n\nYour previous reply was rejected: " + str(exc)
                      + "\nReply again with JSON that follows the required format exactly.")
    raise OutputError(f"{task}: no valid reply after {retries + 1} attempts ({last})")
