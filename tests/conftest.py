import pytest

from josie.bc import Dirichlet
from josie.geom import Line, CircleArc
from josie.mesh import Mesh


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", help="Some tests can plot the mesh. "
        "Set to true if you want to see them"
    )


@pytest.fixture
def boundaries():
    left = Line([0, 0], [0, 1])
    bottom = CircleArc([0, 0], [1, 0], [0.2, 0.2])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    bc = Dirichlet(0)

    left.bc = bc
    bottom.bc = bc
    right.bc = bc
    top.bc = bc

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top)
    mesh.interpolate(20, 20)
    mesh.generate()

    yield mesh


@pytest.fixture
def plot(request):
    return request.config.getoption("--plot")


@pytest.fixture
def tol():
    yield 1E-12
