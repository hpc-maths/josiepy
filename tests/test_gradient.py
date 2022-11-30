# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import pytest

from josie.bc import BoundaryCondition, ScalarBC
from josie.dimension import Dimensionality
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.solver import Solver
from josie.general.schemes.diffusive.lstsq import LeastSquareGradient


@pytest.fixture
def gradient_one_boundaries(boundaries, Q):
    """Change the BCs to enforce the gradient == 1 also on ghost cells"""

    class GradientOneBC(ScalarBC):
        def __call__(self, cells, ghost_cells, field, t):
            num_ghosts_x, num_ghosts_y, _ = ghost_cells.centroids.shape
            return np.sum(ghost_cells.centroids, axis=-1).view(Q)

    left, bottom, right, top = boundaries

    bc = BoundaryCondition(Q(GradientOneBC()))

    left.bc = bc
    right.bc = bc
    top.bc = bc
    bottom.bc = bc

    yield (left, bottom, right, top)


@pytest.fixture(params=(Dimensionality.ONED, Dimensionality.TWOD))
def mesh(request, gradient_one_boundaries):
    left, bottom, right, top = gradient_one_boundaries

    ny = 3

    # Hack for 1D
    if request.param < 2:
        # Remove the curve, for 1D we must have straight lines
        bottom = top
        top.bc = None
        bottom.bc = None
        ny = 1

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(3, ny)
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
        # The np.newaxis to take into account the num_dofs dimension of the
        # array
        cells.values = np.sum(cells.centroids, axis=-1)[..., np.newaxis].view(Q)

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

    scheme.pre_step(mesh.cells, 0)
    scheme.pre_accumulate(mesh.cells, 0, 0)

    assert np.all(np.abs(scheme._gradient - 1) < tol)
