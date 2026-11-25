"""Prompt builders for the decomposed coding steps.

AASL classification is split into separate calls so each decision is made
on its own and each can be evaluated on its own:

1. ``detect``     – is the unit in the theoretical domain D?
2. ``assign``     – which category of C, or the residual?
3. ``subcategory`` – subcategory, for categories that define them.
4. ``attributes`` – attributes of the assigned category.
5. ``propose``    – (discovery) candidate categories from residual units.

Each builder returns the system text, user text and the JSON schema the reply
must satisfy. Schemas avoid nulls (sentinel strings are used instead) so the
same schema works with strict structured-output modes of different providers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import NO_SUBCATEGORY, RESIDUAL, TextUnit
from .schema import Schema

NOT_STATED = "NOT_STATED"


@dataclass(frozen=True)
class Prompt:
    task: str
    system: str
    user: str
    schema: dict


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def _unit_block(unit: TextUnit) -> str:
    return f'<unit id="{unit.uid}">\n{unit.text}\n</unit>'


def _bullets(items) -> str:
    return "\n".join(f"  - {x}" for x in items)


def _domain_text(schema: Schema) -> str:
    d = schema.domain
    parts = [f"Domain: {d.name}", f"Definition: {d.definition}"]
    if d.include:
        parts.append("Counts as in the domain:\n" + _bullets(d.include))
    if d.exclude:
        parts.append("Does not count as in the domain:\n" + _bullets(d.exclude))
    notes = schema.notes_for("domain")
    if notes:
        parts.append("Clarifications:\n" + _bullets(notes))
    return "\n".join(parts)


SYSTEM = ("You code text units for a research project. Apply the definitions exactly as "
          "written. Base every decision only on the text of the unit. When you cite "
          "evidence, copy a short passage word for word from the unit. Reply only with "
          "JSON in the requested format.")


def detect_prompt(schema: Schema, unit: TextUnit) -> Prompt:
    user = "\n\n".join([
        "Decide whether the unit below belongs to the domain.",
        _domain_text(schema),
        _unit_block(unit),
        'Reply with "in_domain" (true or false) and "evidence": the passage that '
        'decided it, copied verbatim, or an empty string if nothing in the unit is relevant.',
    ])
    schema_ = _obj({"in_domain": {"type": "boolean"}, "evidence": {"type": "string"}})
    return Prompt("detect", SYSTEM, user, schema_)


def _category_text(schema: Schema) -> str:
    blocks = []
    for c in schema.categories:
        lines = [f"[{c.name}]", f"Definition: {c.definition}"]
        if c.include:
            lines.append("Includes:\n" + _bullets(c.include))
        if c.exclude:
            lines.append("Excludes:\n" + _bullets(c.exclude))
        if c.examples:
            lines.append("Examples:\n" + _bullets(f'"{e}"' for e in c.examples[:3]))
        notes = schema.notes_for(c.name)
        if notes:
            lines.append("Clarifications:\n" + _bullets(notes))
        blocks.append("\n".join(lines))
    general = schema.notes_for("categories")
    if general:
        blocks.append("General clarifications:\n" + _bullets(general))
    return "\n\n".join(blocks)


def assign_prompt(schema: Schema, unit: TextUnit) -> Prompt:
    user = "\n\n".join([
        f"The unit below has been judged to be in the domain \"{schema.domain.name}\". "
        "Assign it to the one category whose definition it meets.",
        "Categories:\n\n" + _category_text(schema),
        f"If the unit is in the domain but none of the categories fits, answer {RESIDUAL}. "
        f"{RESIDUAL} is a valid and useful answer: do not stretch a definition to avoid it.",
        _unit_block(unit),
        'Reply with "category" and "evidence": the passage that supports the choice, '
        "copied verbatim from the unit.",
    ])
    enum = schema.category_names + [RESIDUAL]
    schema_ = _obj({"category": {"type": "string", "enum": enum}, "evidence": {"type": "string"}})
    return Prompt("assign", SYSTEM, user, schema_)


def subcategory_prompt(schema: Schema, unit: TextUnit, category: str) -> Prompt:
    cat = schema.category(category)
    subs = "\n".join(f"- {sc.name}: {sc.definition}" for sc in cat.subcategories)
    user = "\n\n".join([
        f"The unit below is coded as \"{category}\": {cat.definition}",
        "Choose the subcategory that fits it:\n" + subs,
        f"If none of them fits, answer {NO_SUBCATEGORY}.",
        _unit_block(unit),
    ])
    enum = cat.subcategory_names + [NO_SUBCATEGORY]
    return Prompt("subcategory", SYSTEM, user, _obj({"subcategory": {"type": "string", "enum": enum}}))


def attributes_prompt(schema: Schema, unit: TextUnit, category: str) -> Prompt:
    attrs = schema.attributes_for(category)
    lines, props = [], {}
    for a in attrs:
        if a.kind == "single":
            lines.append(f"- {a.name}: {a.description}. Choose one of: {', '.join(a.options)}; "
                         f"or {NOT_STATED} if the unit does not say.")
            props[a.name] = {"type": "string", "enum": list(a.options) + [NOT_STATED]}
        elif a.kind == "multi":
            lines.append(f"- {a.name}: {a.description}. List every option that applies from: "
                         f"{', '.join(a.options)}. Use an empty list if none is stated.")
            props[a.name] = {"type": "array", "items": {"type": "string", "enum": list(a.options)}}
        else:
            lines.append(f"- {a.name}: {a.description}. Copy each value as written in the unit; "
                         "empty list if none.")
            props[a.name] = {"type": "array", "items": {"type": "string"}}
        for n in schema.notes_for(f"attr:{a.name}"):
            lines.append(f"    clarification: {n}")
    user = "\n\n".join([
        f"The unit below is coded as \"{category}\". Extract these attributes. Only record "
        "what the unit states; do not infer.",
        "\n".join(lines),
        _unit_block(unit),
    ])
    return Prompt("attributes", SYSTEM, user, _obj(props))


def propose_prompt(schema: Schema, units: list, rejected: list = None) -> Prompt:
    """``rejected`` lists earlier proposals the researcher turned down, as dicts with
    ``name``, ``definition`` and ``rationale``; the model is told not to repeat them."""
    existing = "\n".join(f"- {c.name}: {c.definition}" for c in schema.categories)
    parts = [
        f"Every unit below was judged to be in the domain \"{schema.domain.name}\" but was not "
        "assigned to any existing category. A few may have been judged in-domain by mistake; "
        "ignore those.",
        "Existing categories:\n" + existing,
    ]
    if rejected:
        parts.append("Proposals the researcher already rejected (do not propose these again "
                     "unless the units below show a clearly different pattern):\n" + "\n".join(
                         f"- {r['name']}: {r['definition']} (rejected because: {r['rationale']})"
                         for r in rejected))
    user = "\n\n".join(parts + [
        "Propose new categories that would classify these units. Rules:\n"
        "  - a proposal must cover at least two of the units;\n"
        "  - it must not duplicate an existing category;\n"
        "  - give a short name and a one-sentence definition;\n"
        "  - for each unit it covers, give the unit id and a passage copied word for word "
        "from that unit that shows why it belongs.\n"
        "Units that fit no common pattern can be left out. If there is no common pattern, "
        "return an empty list.",
        "\n\n".join(_unit_block(u) for u in units),
    ])
    support = _obj({"unit_id": {"type": "string"}, "quote": {"type": "string"}})
    proposal = _obj({"name": {"type": "string"}, "definition": {"type": "string"},
                     "supporting": {"type": "array", "items": support}})
    schema_ = _obj({"proposals": {"type": "array", "items": proposal}})
    return Prompt("propose", SYSTEM, user, schema_)
