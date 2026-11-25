import pytest

from aasl.schema import Attribute, Category, Domain, Schema


@pytest.fixture
def small_schema():
    return Schema(
        domain=Domain("Cooperation", "Acting with an external organisation toward a shared goal.",
                      include=["joint activity"], exclude=["unilateral action"]),
        categories=[Category("Alpha", "First kind."), Category("Beta", "Second kind.")],
        attributes=[Attribute("form", "Formal basis", "single", ["Contract", "Equity"]),
                    Attribute("partners", "Partner names", "text")])


@pytest.fixture
def sub_schema():
    from aasl.schema import Subcategory
    return Schema(
        domain=Domain("Cooperation", "Acting with an external organisation toward a shared goal."),
        categories=[Category("Alpha", "First kind.",
                             subcategories=[Subcategory("S1", "first sub"), Subcategory("S2", "second sub")]),
                    Category("Beta", "Second kind.")],
        attributes=[Attribute("form", "Formal basis", "single", ["Contract", "Equity"]),
                    Attribute("partners", "Partner names", "text")])
