"""Adapter request shapes, checked against fake provider SDKs (no network)."""
import re
import sys
from types import SimpleNamespace

import pytest

from aasl.llm.base import CachedModel, LanguageModel
from aasl.llm.factory import build_model
from aasl.prompts import (assign_prompt, attributes_prompt, detect_prompt, propose_prompt,
                          subcategory_prompt)
from aasl.core import TextUnit

SCHEMA = {"type": "object", "properties": {"in_domain": {"type": "boolean"},
                                           "evidence": {"type": "string"}},
          "required": ["in_domain", "evidence"], "additionalProperties": False}


@pytest.fixture
def fake_openai(monkeypatch):
    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            msg = SimpleNamespace(content='{"in_domain": true, "evidence": ""}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class OpenAI:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    return calls


@pytest.fixture
def fake_anthropic(monkeypatch):
    calls = []

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            block = SimpleNamespace(type="tool_use", input={"in_domain": False, "evidence": ""})
            return SimpleNamespace(content=[block])

    class Anthropic:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Anthropic))
    return calls


def test_openai_requests_use_strict_json_schema(fake_openai):
    from aasl.llm.openai_model import OpenAIModel
    m = OpenAIModel("gpt-test", temperature=0.0, seed=3)
    assert m.generate("sys", "usr", SCHEMA, "detect") == '{"in_domain": true, "evidence": ""}'
    kw = fake_openai[0]
    js = kw["response_format"]["json_schema"]
    assert js["strict"] is True and js["name"] == "detect" and js["schema"] is SCHEMA
    assert kw["temperature"] == 0.0 and kw["seed"] == 3
    assert [x["role"] for x in kw["messages"]] == ["system", "user"]
    reasoning = OpenAIModel("o-test", temperature=None)
    reasoning.generate("s", "u", SCHEMA, "detect")
    assert "temperature" not in fake_openai[1] and "seed" not in fake_openai[1]
    assert m.name != reasoning.name and "t=0.0" in m.name and "seed=3" in m.name


def test_anthropic_requests_force_one_tool_call(fake_anthropic):
    import json
    from aasl.llm.anthropic_model import AnthropicModel
    m = AnthropicModel("claude-test", temperature=0.0)
    out = m.generate("sys", "usr", SCHEMA, "detect")
    assert json.loads(out) == {"in_domain": False, "evidence": ""}
    kw = fake_anthropic[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "record_detect"}
    assert kw["tools"][0]["input_schema"] is SCHEMA and kw["temperature"] == 0.0
    AnthropicModel("claude-test", temperature=None).generate("s", "u", SCHEMA, "detect")
    assert "temperature" not in fake_anthropic[1]


def test_factory_validation_and_cache_wrapping(fake_openai, tmp_path):
    with pytest.raises(ValueError, match="model.name"):
        build_model({"provider": "openai"})
    with pytest.raises(ValueError, match="unknown model provider"):
        build_model({"provider": "other", "name": "x"})
    m = build_model({"provider": "openai", "name": "gpt-test"}, cache_dir=tmp_path / "c")
    assert isinstance(m, CachedModel)


def test_cache_is_not_shared_between_settings(tmp_path):
    class Fixed(LanguageModel):
        def __init__(self, name):
            self.name = name
            self.calls = 0

        def generate(self, system, user, schema, task):
            self.calls += 1
            return self.name

    cold, hot = Fixed("openai:m|t=0.0|seed=None"), Fixed("openai:m|t=1.0|seed=None")
    assert CachedModel(cold, tmp_path).generate("s", "u", {}, "detect") == cold.name
    assert CachedModel(hot, tmp_path).generate("s", "u", {}, "detect") == hot.name   # not cold's reply
    assert hot.calls == 1


def test_every_prompt_schema_meets_openai_strict_mode_rules(sub_schema):
    u = TextUnit("u", "text")
    prompts = [detect_prompt(sub_schema, u), assign_prompt(sub_schema, u),
               subcategory_prompt(sub_schema, u, "Alpha"),
               attributes_prompt(sub_schema, u, "Alpha"), propose_prompt(sub_schema, [u, u])]

    def check(spec, path="$"):
        problems = []
        if spec.get("type") == "object":
            if spec.get("additionalProperties") is not False:
                problems.append(f"{path}: object not closed")
            if set(spec.get("required", [])) != set(spec["properties"]):
                problems.append(f"{path}: not every property is required")
            for k, v in spec["properties"].items():
                problems += check(v, f"{path}.{k}")
        if spec.get("type") == "array":
            problems += check(spec["items"], path + "[]")
        return problems

    for p in prompts:
        assert check(p.schema) == [], p.task
        assert re.fullmatch(r"[A-Za-z0-9_-]+", p.task)          # allowed as a schema/tool name
