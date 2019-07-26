import numpy as np


def test_neigh_state(solver):
    cell = solver.mesh.cells[0, 0]
    east_neigh = solver.mesh.cells[1, 0]

    assert cell.e.value == east_neigh.value

    east_neigh.value = solver.problem.Q(14)

    assert cell.e.value == east_neigh.value


def test_neighbours_len(solver):
    """ Testing that in the 2D case each cells has four neighbours """

    for cell in solver.mesh.cells.ravel():
        assert len(list(iter(cell))) == 4


def test_init(solver):

    def assert_init(cell):
        xc, _ = cell.centroid

        if xc > 0.45:
            assert np.array_equal(cell.value, solver.problem.Q(1))
        else:
            assert np.array_equal(cell.value, solver.problem.Q(0))

    for cell in solver.mesh.cells.ravel():
        assert_init(cell)
