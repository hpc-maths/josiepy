import numpy as np


def test_init(solver):

    xc = solver.mesh.centroids[:, :0]

    assert np.all(solver.values[np.where(xc >= 0.45), :, :] == solver.Q(1))
    assert np.all(solver.values[np.where(xc < 0.45), :, :] == solver.Q(0))


def test_write(tmp_path, solver):
    # TODO: Think about a better test. E.g. Write down, then reload, and
    # compare

    file = tmp_path / "test_write.xdmf"

    solver.save(0, file.as_posix())
