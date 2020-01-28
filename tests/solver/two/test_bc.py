import numpy as np


def test_periodic(solver):
    """ Testing that the neighbours of boundary cells are the corresponding
    cells on the other domain and, for 1D, that the cells on the 'top' and
    'bottom' boundary actually do not have neighbours
    """
    assert np.array_equal(solver.left_ghost, solver.values[-1, :])
    assert np.array_equal(solver.right_ghost, solver.values[0, :])
    assert np.array_equal(solver.top_ghost, solver.values[:, 0])
    assert np.array_equal(solver.btm_ghost, solver.values[:, -1])


def test_periodic_state(solver):
    """ Just testing that updating the values of the cells that are interlinked
    as periodic, we get the actual state and not the previous one """

    # The actual values are just random chosen
    a = 1.11
    b = 2.22
    c = 3.33
    d = 4.44

    solver.values[0, :] = a
    solver._update_ghosts()
    assert np.all(solver.right_ghost == a)

    solver.values[-1, :] = b
    solver._update_ghosts()
    assert np.all(solver.left_ghost == b)

    solver.values[:, 0] = c
    solver._update_ghosts()
    assert np.all(solver.top_ghost == c)

    solver.values[:, -1] = d
    solver._update_ghosts()
    assert np.all(solver.btm_ghost == d)
