"""Build a model from the ``model`` section of a config."""
from __future__ import annotations

from .base import CachedModel, LanguageModel


def build_model(cfg: dict, cache_dir=None) -> LanguageModel:
    provider = cfg.get("provider", "openai").lower()
    name = cfg.get("name")
    if not name:
        raise ValueError("model.name is required; pin a dated snapshot")
    if provider == "openai":
        from .openai_model import OpenAIModel
        model = OpenAIModel(name, temperature=cfg.get("temperature", 0.0), seed=cfg.get("seed"))
    elif provider == "anthropic":
        from .anthropic_model import AnthropicModel
        model = AnthropicModel(name, temperature=cfg.get("temperature", 0.0),
                               max_tokens=cfg.get("max_tokens", 2048))
    else:
        raise ValueError(f"unknown model provider {provider!r} (openai | anthropic)")
    return CachedModel(model, cache_dir) if cache_dir else model
