import pytest

from josie.bc import make_periodic, Direction
from josie.geom import Line
from josie.mesh import Mesh, SimpleCell
from josie.solver import Solver


@pytest.fixture
def boundaries():
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(20, 20)
    mesh.generate()

    yield mesh


@pytest.fixture
def solver(mocker, mesh, Q, init_fun):
    """ A dummy solver instance with initiated state """

    scheme = mocker.Mock()

    solver = Solver(mesh, Q, scheme)
    solver.init(init_fun)

    yield solver
