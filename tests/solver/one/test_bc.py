import pytest

from josie.mesh.mesh import Mesh
from josie.mesh.cell import NeighbourCell
from josie.exceptions import InvalidMesh


def test_invalid_mesh_x(boundaries, solver, init_fun):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top)
    with pytest.raises(InvalidMesh):
        mesh.interpolate(40, 2)


def test_periodic(solver):
    """ Testing that the neighbours of boundary cells are the corresponding
    cells on the other domain and, for 1D, that the cells on the 'top' and
    'bottom' boundary actually do not have neighbours
    """

    for left_cell in solver.mesh.cells[0, :]:
        assert isinstance(left_cell.w, NeighbourCell)

    for btm_cell in solver.mesh.cells[:, 0]:
        assert btm_cell.s is None

    for right_cell in solver.mesh.cells[-1, :]:
        assert isinstance(right_cell.e, NeighbourCell)

    for top_cell in solver.mesh.cells[:, -1]:
        assert top_cell.n is None


def test_periodic_state(solver):
    """ Just testing that updating the values of the cells that are interlinked
    as periodic, we get the actual state and not the previous one """

    first_cell = solver.mesh.cells[0, 0]
    first_cell.value = 42.19

    last_cell = solver.mesh.cells[-1, 0]
    last_cell.value = 19.42

    assert last_cell.e.value == 42.19
    assert first_cell.w.value == 19.42
