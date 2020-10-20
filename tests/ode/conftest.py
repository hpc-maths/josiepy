import pytest

from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell


@pytest.fixture
def boundaries(request):
    """ 0D problem along x """
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left.bc = None
    right.bc = None
    top.bc = None
    bottom.bc = None

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    # Just one cell space for time-only
    mesh.interpolate(1, 1)
    mesh.generate()

    yield mesh
