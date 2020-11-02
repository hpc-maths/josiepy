import numpy as np


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
    assert np.array_equal(
        solver.mesh.cells._values[1:-1, -1], solver.mesh.cells.values[:, 0]
    )
    assert np.array_equal(
        solver.mesh.cells._values[1:-1, 0], solver.mesh.cells.values[:, -1]
    )


def test_periodic_state(solver):
    """Just testing that updating the values of the cells that are interlinked
    as periodic, we get the actual state and not the previous one"""

    # The actual values are just random chosen
    a = 1.11
    b = 2.22
    c = 3.33
    d = 4.44

    solver.mesh.cells.values[0, :] = a
    solver.mesh.update_ghosts(t=0)
    assert np.all(solver.mesh.cells._values[-1, 1:-1] == a)

    solver.mesh.cells.values[-1, :] = b
    solver.mesh.update_ghosts(t=0)
    assert np.all(solver.mesh.cells._values[0, 1:-1] == b)

    solver.mesh.cells.values[:, 0] = c
    solver.mesh.update_ghosts(t=0)
    assert np.all(solver.mesh.cells._values[1:-1, -1] == c)

    solver.mesh.cells.values[:, -1] = d
    solver.mesh.update_ghosts(t=0)
    assert np.all(solver.mesh.cells._values[1:-1, 0] == d)
