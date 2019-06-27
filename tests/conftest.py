import pytest


@pytest.fixture()
def tol():
    yield 1E-12
