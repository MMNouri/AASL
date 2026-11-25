"""OpenAI adapter using structured outputs (JSON schema, strict mode).

Requires ``pip install openai`` and ``OPENAI_API_KEY`` in the environment.
Pin a dated model snapshot in the config so a long run cannot drift. Set
``temperature: null`` for models that do not accept a temperature.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import LanguageModel


class OpenAIModel(LanguageModel):
    def __init__(self, model: str, temperature: Optional[float] = 0.0, seed: Optional[int] = None,
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 60.0):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install the 'openai' extra: pip install aasl[openai]") from exc
        # The name is part of the response-cache key, so every setting that changes
        # replies must appear in it.
        self.name = f"openai:{model}|t={temperature}|seed={seed}"
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.client = OpenAI(api_key=os.environ.get(api_key_env), timeout=timeout)

    def generate(self, system: str, user: str, schema: dict, task: str) -> str:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": task, "schema": schema, "strict": True}},
        )
        if self.temperature is not None:        # some reasoning models reject the parameter
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
