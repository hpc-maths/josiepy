import pytest

from josie.bc import make_periodic, Direction
from josie.geom import CircleArc, Line
from josie.mesh import Mesh
from josie.mesh import SimpleCell


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        help="Some tests can plot the mesh. "
        "Set to true if you want to see them",
    )

    parser.addoption(
        "--write",
        action="store_true",
        help="Some tests can output to disk data. Set to true if you "
        "want to write the data on disk",
    )

    parser.addoption(
        "--bench",
        action="store_true",
        help="Run benchmarks instead of unit tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--bench"):
        skip_reason = pytest.mark.skip(
            reason="Running Benchmarks, not unit tests"
        )
        for item in items:
            if not ("bench") in item.keywords:
                item.add_marker(skip_reason)
    else:
        skip_reason = pytest.mark.skip(reason="Skipping benchmarks")

        for item in items:
            if "bench" in item.keywords:
                item.add_marker(skip_reason)


@pytest.fixture(
    params=(Line([0, 0], [1, 0]), CircleArc([0, 0], [1, 0], [0.2, 0.2]))
)
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

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(3, 3)
    mesh.generate()

    yield mesh


@pytest.fixture(scope="session", autouse=True)
def plot(request):
    if not (request.config.getoption("--plot")):
        import matplotlib

        matplotlib.use("SVG")
    yield request.config.getoption("--plot")


@pytest.fixture(scope="session", autouse=True)
def write(request):
    yield request.config.getoption("--write")


@pytest.fixture
def tol():
    yield 1e-12
