# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import pytest


from josie.general.schemes.time import ExplicitEuler
from josie.general.schemes.space.muscl import MUSCL_Hancock
from josie.general.schemes.space.limiters import No_Limiter

from josie.mesh.cellset import MeshCellSet
from josie.solver import Solver
from josie.mesh import Mesh
from josie.mesh.cell import MUSCLCell

from josie.advection.state import Q
from josie.advection.schemes import Upwind
from josie.advection.problem import AdvectionProblem

# Advection velocity in x-direction
V = np.array([1.0, 0.0])


@pytest.fixture(
    params=[-1.0, 0, 0.1, 1.0],
)
def omega(request):
    yield request.param


class Scheme(Upwind, MUSCL_Hancock, No_Limiter, ExplicitEuler):
    pass


def test_order_muscl_hancock(plot, omega, boundaries, init):
    """Testing against the real 1D solution"""

    L2_err = []
    nx_tab = [30, 50, 100, 300, 500, 1000]
    plt.figure()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., [0]]

        cells.values = init(np.array(xc)).view(Q)

    for nx in nx_tab:
        left, bottom, right, top = boundaries

        mesh = Mesh(left, bottom, right, top, MUSCLCell)
        mesh.interpolate(nx, 1)
        mesh.generate()

        scheme = Scheme(AdvectionProblem(V))
        scheme.omega = omega

        solver = Solver(mesh, Q, scheme)
        solver.init(init_fun)

        # CFL condition
        c = 0.5
        dx = 1 / nx
        dt = c * dx
        T = 0.1

        x = solver.mesh.cells.centroids[..., 0, 0]
        x = x.reshape(x.size)
        Nt = int(np.ceil(T / dt))
        for t in np.linspace(0, Nt * dt, Nt + 1):
            u = solver.mesh.cells.values[..., 0, 0]
            u = u.reshape(u.size)

            err = u - init(x - t)

            solver.step(dt)

        L2_err.append(np.linalg.norm(err) * np.sqrt(dx))

    if plot:
        plt.loglog(
            nx_tab,
            L2_err[-1] * nx_tab[-1] / np.array(nx_tab),
            "--",
            label=r"$\propto \Delta x$",
        )
        plt.loglog(
            nx_tab,
            L2_err[-1] * nx_tab[-1] ** 2 / np.array(nx_tab) ** 2,
            "--",
            label=r"$\propto \Delta x^2$",
        )
        plt.loglog(
            nx_tab,
            L2_err[-1] * nx_tab[-1] ** 3 / np.array(nx_tab) ** 3,
            "--",
            label=r"$\propto \Delta x^3$",
        )
        plt.scatter(nx_tab, np.array(L2_err), label=r"$E_{L^2}$")
        plt.xlabel(r"$\frac{1}{\Delta x}$")
        plt.ylabel(r"$L^2$ error")
        plt.title(r"L2 error for $\omega=$" + str(omega))
        plt.legend(loc="lower left")

        plt.show()

    eps = 0.2
    order = -np.linalg.lstsq(
        np.vstack([np.log(nx_tab), np.ones(len(nx_tab))]).T,
        np.log(L2_err),
        rcond=None,
    )[0][0]
    print(order)

    assert (order > 2 - eps and not (scheme.omega == 1 / 3 * (2 * c - np.sign(c)))) or (
        order > 3 - eps and scheme.omega == 1 / 3 * (2 * c - np.sign(c))
    )
