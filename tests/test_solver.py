# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.solver import Solver


def test_init(solver):
    xc = solver.mesh.cells.centroids[..., 0]

    xc_r = np.where(xc >= 0.45)
    xc_l = np.where(xc < 0.45)

    right_values = solver.mesh.cells.values[xc_r[0], xc_r[1], :]
    left_values = solver.mesh.cells.values[xc_l[0], xc_l[1], :]

    assert np.all(right_values == 1.0)
    assert np.all(left_values == 0.0)


def test_plot(solver):
    solver.plot()
    solver.show("u")


def test_animate(solver):
    for i in range(10):
        solver.animate(i)
        solver.mesh.cells.values[1, :, :] -= 1

    solver.show("u")


def test_linear_index(mocker, mesh, Q):
    scheme = mocker.Mock()
    solver = Solver(mesh, Q, scheme)
    _values = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ],
        dtype=float,
    )

    def init_fun(solver: Solver):

        solver._values = _values.T[:, :, np.newaxis]

    solver.init(init_fun)  # This also updates ghosts

    assert np.all(
        solver.mesh.cells.values[0, :].ravel() == np.array([6, 11, 16]).ravel()
    )
    assert np.all(
        solver.mesh.cells.values[:, 1].ravel()
        == np.array([11, 12, 13]).ravel()
    )

    # This got updated after init
    assert np.all(
        solver.mesh.cells._values[0, 1:-1].ravel()
        == np.array([8, 13, 18]).ravel()
    )
