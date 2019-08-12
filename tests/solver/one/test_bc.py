import numpy as np
import pytest

from josie.mesh.mesh import Mesh
from josie.exceptions import InvalidMesh


def test_invalid_mesh_x(boundaries, solver, init_fun):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top)
    with pytest.raises(InvalidMesh):
        mesh.interpolate(40, 2)


def test_1D(solver):
    """ In 1D, the top and bottom ghost cells are not present """

    with pytest.raises(AttributeError):
        solver.top_ghost
        solver.btm_ghost


def test_periodic(solver):
    """ Testing that the neighbours of boundary cells are the corresponding
    cells on the other domain and, for 1D, that the cells on the 'top' and
    'bottom' boundary actually do not have neighbours
    """

    assert np.array_equal(solver.left_ghost, solver.values[-1, :])
    assert np.array_equal(solver.right_ghost, solver.values[0, :])


def test_periodic_state(solver):
    """ Just testing that updating the values of the cells that are interlinked
    as periodic, we get the actual state and not the previous one """

    solver.values[0, 0] = 42.19

    assert solver.right_ghost[0, 0] == 42.19
