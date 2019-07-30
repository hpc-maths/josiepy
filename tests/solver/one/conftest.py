import pytest

from josie.bc import make_periodic, Direction
from josie.geom import Line
from josie.mesh import Mesh


@pytest.fixture
def boundaries():
    """ 1D problem along x """
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left, right = make_periodic(left, right, Direction.X)
    top.bc = None
    bottom.bc = None

    yield (left, bottom, right, top)


@pytest.fixture
def boundaries_y():
    """ 1D problem along y """
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    bottom, top = make_periodic(bottom, top, Direction.X)
    right.bc = None
    left.bc = None

    yield (left, bottom, right, top)


@pytest.fixture()
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top)
    mesh.interpolate(40, 1)
    mesh.generate()

    yield mesh