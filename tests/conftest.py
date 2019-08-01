import pytest

from josie.bc import make_periodic, Direction
from josie.geom import CircleArc, Line
from josie.mesh import Mesh


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", help="Some tests can plot the mesh. "
        "Set to true if you want to see them"
    )


@pytest.fixture(params=(
    Line([0, 0], [1, 0]),
    CircleArc([0, 0], [1, 0], [0.2, 0.2])
))
def boundaries(request):
    left = Line([0, 0], [0, 1])
    bottom = request.param
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top)
    mesh.interpolate(4, 4)
    mesh.generate()

    yield mesh


@pytest.fixture(scope='session', autouse=True)
def plot(request):
    if not(request.config.getoption("--plot")):
        import matplotlib
        matplotlib.use('Template')
    yield request.config.getoption("--plot")


@pytest.fixture
def tol():
    yield 1E-12
