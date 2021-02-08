import pytest

from josie.solver.fields import Fields


@pytest.fixture
def fields():
    class NewFields(Fields):
        a = 0
        b = 1

    yield NewFields


def test_functional_api():
    fields = Fields("Test", {"a": 0, "b": 1})

    assert fields.a == 0
    assert fields.b == 1
    assert len(fields) == 2


def test_no_init():
    with pytest.raises(TypeError):
        Fields()


def test_int_type(fields):
    for f in fields:
        assert isinstance(f, int)


def test_field_name(fields):
    assert fields.a.name == "a"
    assert fields.b.name == "b"


def test_field_value(fields):
    assert fields.a.value == 0
    assert fields.b.value == 1
