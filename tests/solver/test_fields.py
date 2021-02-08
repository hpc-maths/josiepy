import pytest

from josie.solver.fields import Fields


def test_functional_api():
    fields = Fields("Test", {"a": 0, "b": 1})
    assert fields.a == 0
    assert fields.b == 1
    assert len(fields) == 2


def test_no_init():
    with pytest.raises(TypeError):
        Fields()
