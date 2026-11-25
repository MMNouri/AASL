import pytest

from aasl.parsing import OutputError, extract_json, parse_reply, validate

SCHEMA = {"type": "object", "properties": {
    "category": {"type": "string", "enum": ["A", "RESIDUAL"]},
    "tags": {"type": "array", "items": {"type": "string", "enum": ["x", "y"]}},
    "flag": {"type": "boolean"}},
    "required": ["category", "tags", "flag"], "additionalProperties": False}


def test_extract_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure: {"a": 2} done') == {"a": 2}
    with pytest.raises(OutputError):
        extract_json("no json here")


def test_valid_reply_passes():
    assert parse_reply('{"category": "A", "tags": ["x"], "flag": true}', SCHEMA)["category"] == "A"


@pytest.mark.parametrize("raw", [
    '{"category": "Z", "tags": [], "flag": true}',          # label outside the enum
    '{"category": "A", "tags": ["q"], "flag": true}',       # bad list item
    '{"category": "A", "tags": []}',                         # missing field
    '{"category": "A", "tags": [], "flag": 1}',              # wrong type
    '{"category": "A", "tags": [], "flag": true, "x": 1}',   # extra field
])
def test_invalid_replies_rejected(raw):
    with pytest.raises(OutputError):
        parse_reply(raw, SCHEMA)


def test_bool_is_not_a_number():
    assert validate(True, {"type": "integer"})
