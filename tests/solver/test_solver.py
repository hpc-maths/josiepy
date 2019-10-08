import numpy as np


def test_init(solver):
    xc = solver.mesh.centroids[:, :, 0]

    xc_r = np.where(xc >= 0.45)
    xc_l = np.where(xc < 0.45)

    right_values = solver.values[xc_r[0], xc_r[1], :]
    left_values = solver.values[xc_l[0], xc_l[1], :]

    assert np.all(right_values == solver.Q(1))
    assert np.all(left_values == solver.Q(0))


def test_save(tmp_path, solver):
    # TODO: Think about a better test. E.g. Write down, then reload, and
    # compare

    file = tmp_path / "test_write.xdmf"

    solver.save(0, file.as_posix())


def test_plot(solver):
    solver.plot()
    solver.show("u")


def test_animate(solver):
    for i in range(10):
        solver.animate(i)
        solver.values[i, :, :] -= 1

    solver.show("u")
