import numpy as np

import pytest

from josie.bc import BoundaryCondition, ScalarBC
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.general.schemes.diffusive import LeastSquareGradient
from josie.solver import Solver


@pytest.fixture
def gradient_one_boundaries(boundaries, Q):
    """ Change the BCs to enforce the gradient == 1 also on ghost cells """

    class GradientOneBC(ScalarBC):
        def __call__(self, cells, ghost_centroids, t):
            return np.sum(ghost_centroids, axis=-1).view(Q)

    left, bottom, right, top = boundaries

    left.bc = bottom.bc = right.bc = top.bc = BoundaryCondition(
        Q(GradientOneBC())
    )

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(gradient_one_boundaries):
    left, bottom, right, top = gradient_one_boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(3, 3)
    mesh.generate()

    yield mesh


@pytest.fixture
def init_fun(Q):
    r"""Init the state in order to have gradient == 1 in all the cells, in all
    directions

    .. math::

        \phi\qty(\vb{x}) = x + y

    """

    def init_fun(cells: MeshCellSet):
        cells.values = np.sum(cells.centroids, axis=-1).view(Q)

    yield init_fun


def test_least_square_gradient(mocker, tol, mesh, Q, init_fun):
    """Test that calculating the gradient on a field having unitary gradient
    components actually returns 1 for all components on all cells"""

    # Patch LeastSquareGradient to allow init even if it's an ABC
    mocker.patch.object(LeastSquareGradient, "__abstractmethods__", set())
    problem = mocker.Mock()

    scheme = LeastSquareGradient(problem)

    solver = Solver(mesh, Q, scheme)
    solver.init(init_fun)

    scheme.pre_step(solver.mesh.cells, solver.neighbours)

    assert np.all(np.abs(scheme._gradient - 1 < tol))
