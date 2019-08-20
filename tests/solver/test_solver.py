import numpy as np


def test_init(solver):

    xc = solver.mesh.centroids[:, :0]

    assert np.all(solver.values[np.where(xc >= 0.45), :, :] == solver.Q(1))
    assert np.all(solver.values[np.where(xc < 0.45), :, :] == solver.Q(0))
