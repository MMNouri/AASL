import pytest

from aasl.core import RESIDUAL
from aasl.schema import Attribute, Category, Domain, Schema


def test_roundtrip_and_fingerprint(tmp_path, small_schema):
    p = tmp_path / "s.yaml"
    small_schema.save(p)
    back = Schema.load(p)
    assert back.to_dict() == small_schema.to_dict()
    assert back.fingerprint() == small_schema.fingerprint()


def test_reserved_and_duplicate_names_rejected():
    d = Domain("D", "def")
    with pytest.raises(ValueError):
        Schema(d, [Category(RESIDUAL, "x")])
    with pytest.raises(ValueError):
        Schema(d, [Category("A", "x"), Category("A", "y")])
    with pytest.raises(ValueError):
        Attribute("a", "b", "single", [])


def test_versioning_and_change_types(small_schema):
    s2 = small_schema.with_notes({"Alpha": ["clarify"]}, reason="cal")
    assert s2.version == small_schema.version + 1
    assert s2.fingerprint() != small_schema.fingerprint()
    assert s2.category_names == small_schema.category_names
    s3 = s2.with_category(Category("Gamma", "Third.", origin="discovered"), reason="kept")
    assert s3.category_names[-1] == "Gamma" and s3.history[-1]["event"] == "category_added"
    with pytest.raises(ValueError):
        s3.with_category(Category("Gamma", "dup"))
    s4 = s3.with_merge("Alpha", include=["also this"], examples=["q"], reason="merge")
    assert "also this" in s4.category("Alpha").include
    # earlier versions are untouched
    assert "also this" not in s3.category("Alpha").include


def test_attribute_scope(small_schema):
    s = Schema(small_schema.domain, small_schema.categories,
               [Attribute("only_a", "x", "text", applies_to=["Alpha"])])
    assert [a.name for a in s.attributes_for("Alpha")] == ["only_a"]
    assert s.attributes_for("Beta") == []
    with pytest.raises(ValueError):
        Schema(small_schema.domain, small_schema.categories,
               [Attribute("bad", "x", "text", applies_to=["Nope"])])


def test_codebook_mentions_residual(small_schema):
    md = small_schema.to_markdown()
    assert "RESIDUAL" in md and "Alpha" in md


def test_subcategories_roundtrip_validate_and_fingerprint(tmp_path, sub_schema):
    from aasl.schema import Category, Subcategory
    p = tmp_path / "s.yaml"
    sub_schema.save(p)
    back = Schema.load(p)
    assert back.category("Alpha").subcategory_names == ["S1", "S2"]
    assert back.fingerprint() == sub_schema.fingerprint()
    assert "S1: first sub" in sub_schema.to_markdown()
    changed = Schema.from_dict(sub_schema.to_dict())
    changed.category("Alpha").subcategories[0].definition = "different"
    assert changed.fingerprint() != sub_schema.fingerprint()
    with pytest.raises(ValueError):
        Category("X", "d", subcategories=[Subcategory("a", "1"), Subcategory("a", "2")])
    with pytest.raises(ValueError):
        Category("X", "d", subcategories=[Subcategory("NONE_FITS", "1")])
