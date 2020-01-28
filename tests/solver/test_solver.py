import numpy as np

from josie.solver import Solver


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
        solver.values[1, :, :] -= 1

    solver.show("u")


def test_linear_index(mesh, Q):
    solver = Solver(mesh, Q)
    _values = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ]
    )

    def init_fun(solver: Solver):

        solver._values[:, :] = _values[np.newaxis, :, :].T

    solver.init(init_fun)

    assert np.all(solver.values[0, :].ravel() == np.array([6, 11, 16]).ravel())
    assert np.all(
        solver.values[:, 1].ravel() == np.array([11, 12, 13]).ravel()
    )
    assert np.all(
        solver._values[0, 1:-1].ravel() == np.array([5, 10, 15]).ravel()
    )
