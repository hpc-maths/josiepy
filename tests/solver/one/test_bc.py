# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from josie.mesh.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.exceptions import InvalidMesh


def test_invalid_mesh_x(boundaries, solver, init_fun):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    with pytest.raises(InvalidMesh):
        mesh.interpolate(40, 2)


def test_1D(solver):
    """ In 1D, the top and bottom ghost cells are not present """

    assert solver.mesh.dimensionality == 1
    assert len(solver.mesh.boundaries) == 2


def test_periodic(solver):
    """Testing that the neighbours of boundary cells are the corresponding
    cells on the other domain and, for 1D, that the cells on the 'top' and
    'bottom' boundary actually do not have neighbours
    """

    assert np.array_equal(
        solver.mesh.cells._values[0, 1:-1], solver.mesh.cells.values[-1, :]
    )
    assert np.array_equal(
        solver.mesh.cells._values[-1, 1:-1], solver.mesh.cells.values[0, :]
    )


def test_periodic_state(solver):
    """Just testing that updating the values of the cells that are interlinked
    as periodic, we get the actual state and not the previous one"""
    solver.mesh.cells.values[0, 0] = 42.19
    solver.mesh.update_ghosts(t=0)

    assert solver.mesh.cells._values[-1, 1] == 42.19
