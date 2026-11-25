"""Parse and validate model output.

Model replies are checked against the same JSON schema that was sent with the
request. Anything outside the schema (a label not in the enum, a missing
field) is an error: it is retried and, if it persists, recorded as a coding
error rather than silently mapped to a nearby value.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "number": (int, float), "integer": int, "null": type(None),
}


class OutputError(ValueError):
    """The model reply is not valid JSON for the requested schema."""


def extract_json(raw: str) -> Any:
    """Load the JSON object in ``raw``, tolerating code fences or leading text."""
    if raw is None:
        raise OutputError("empty reply")
    text = _FENCE.sub("", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise OutputError("no JSON object in reply")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise OutputError(f"invalid JSON: {exc}") from exc


def _type_ok(value: Any, typ) -> bool:
    types = typ if isinstance(typ, list) else [typ]
    for t in types:
        py = _TYPES[t]
        if t in ("number", "integer") and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def validate(value: Any, schema: dict, path: str = "$") -> list:
    """Return a list of problems (empty when ``value`` conforms)."""
    problems = []
    if "type" in schema and not _type_ok(value, schema["type"]):
        return [f"{path}: expected {schema['type']}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}: missing {key!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    problems.append(f"{path}: unexpected field {key!r}")
        for key, sub in props.items():
            if key in value:
                problems += validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            problems += validate(item, schema["items"], f"{path}[{i}]")
    return problems


def parse_reply(raw: str, schema: dict) -> dict:
    data = extract_json(raw)
    problems = validate(data, schema)
    if problems:
        raise OutputError("; ".join(problems[:5]))
    return data
