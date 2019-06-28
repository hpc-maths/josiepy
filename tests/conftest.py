import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", help="Some tests can plot the mesh. "
        "Set to true if you want to see them"
    )


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture
def tol():
    yield 1E-12
