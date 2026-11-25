"""Anthropic adapter. The reply schema is enforced by forcing a single tool
call whose input schema is the reply schema; the tool input is returned as
JSON text.

Requires ``pip install anthropic`` and ``ANTHROPIC_API_KEY``. Use a pinned
model identifier from the provider documentation in the config.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .base import LanguageModel


class AnthropicModel(LanguageModel):
    def __init__(self, model: str, temperature: Optional[float] = 0.0, max_tokens: int = 2048,
                 api_key_env: str = "ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("install the 'anthropic' extra: pip install aasl[anthropic]") from exc
        self.name = f"anthropic:{model}|t={temperature}|max={max_tokens}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=os.environ.get(api_key_env))

    def generate(self, system: str, user: str, schema: dict, task: str) -> str:
        tool = {"name": f"record_{task}", "description": f"Record the {task} result.",
                "input_schema": schema}
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}], tools=[tool],
                      tool_choice={"type": "tool", "name": tool["name"]})
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input)
        return "".join(getattr(b, "text", "") for b in resp.content)
