# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import pytest


from josie.general.schemes.time import ExplicitEuler
from josie.general.schemes.space.muscl import MUSCL_Hancock
from josie.general.schemes.space.limiters import No_Limiter

from josie.dimension import MAX_DIMENSIONALITY
from josie.bc import Dirichlet
from josie.mesh.cellset import MeshCellSet
from josie.state import SubsetState, State
from josie.fluid.state import ConsState
from josie.solver import Solver
from josie.problem import Problem
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.boundary import Line

# Advection velocity in x-direction
V = np.array([1.0, 0.0])


def init(x: np.ndarray):
    """Init function."""
    u = np.empty(x.shape)
    a = 0.05
    b = 0.85

    u = (
        (
            np.exp(-1 / (x - a) ** 2)
            * np.exp(-1 / (x - b) ** 2)
            / (
                np.exp(-1 / ((a + b) / 2 - a) ** 2)
                * np.exp(-1 / ((a + b) / 2 - b) ** 2)
            )
        )
        * (x > a)
        * (x < b)
        * (b - a)
    )
    return u


class AdvectionConsState(SubsetState):
    full_state_fields = State.list_to_enum(["u"])
    fields = State.list_to_enum(["u"])  # type: ignore


class Q(ConsState):
    fields = State.list_to_enum(["u"])  # type: ignore
    cons_state = AdvectionConsState


def flux(state_array: Q) -> np.ndarray:
    return np.einsum("j,...i->...ij", V, state_array)


class AdvectionProblem(Problem):
    def F(self, state_array: Q) -> np.ndarray:
        # I multiply each element of the given state array by the velocity
        # vector. I obtain an Nx2 array where each row is the flux on each
        # cell
        return flux(state_array)


@pytest.fixture(
    params=[-1.0, 0, 0.1, 1.0],
)
def omega(request):
    yield request.param


@pytest.fixture
def scheme():
    class Upwind(MUSCL_Hancock, No_Limiter, ExplicitEuler):
        def intercellFlux(
            self, Q_L: Q, Q_R: Q, normals: np.ndarray, surfaces: np.ndarray
        ):
            nx, ny, num_dofs, num_fields = Q_L.shape

            FS = np.zeros_like(Q_L)
            F = np.zeros((nx, ny, num_dofs, num_fields, MAX_DIMENSIONALITY))

            # Dot product of each normal in `norm` by the advection velocity
            # Equivalent to: un = np.sum(Advection.V*(normals), axis=-1)
            Vn = np.einsum("...k,k->...", normals, V)

            # Check where un > 0
            idx = np.where(Vn > 0)

            if np.any(np.nonzero(idx)):
                F[idx] = flux(Q_L)[idx]

            idx = np.where(Vn < 0)
            if np.any(np.nonzero(idx)):
                F[idx] = flux(Q_R)[idx]

            FS = (
                np.einsum("...mkl,...l->...mk", F, normals)
                * surfaces[..., np.newaxis, np.newaxis]
            )

            return FS

        def CFL(
            self,
            cells: MeshCellSet,
            CFL_value: float,
        ) -> float:
            U_abs = np.linalg.norm(V)
            dx = np.min(cells.surfaces)

            return CFL_value * dx / U_abs

    yield Upwind(AdvectionProblem())


def init_fun(cells: MeshCellSet):
    xc = cells.centroids[..., [0]]

    cells.values = init(np.array(xc)).view(Q)


def test_against_real_1D(solver, plot, tol, scheme, omega):
    """Testing against the real 1D solution"""

    L2_err = []
    nx_tab = [30, 50, 100, 300, 500, 1000]
    plt.figure()

    for nx in nx_tab:
        left = Line([0, 0], [0, 1])
        bottom = Line([0, 0], [1, 0])
        right = Line([1, 0], [1, 1])
        top = Line([0, 1], [1, 1])

        left.bc = Dirichlet(AdvectionConsState(0))
        right.bc = Dirichlet(AdvectionConsState(0))
        top.bc = None
        bottom.bc = None

        mesh = Mesh(left, bottom, right, top, SimpleCell)
        mesh.interpolate(nx, 1)
        mesh.generate()

        musclScheme = scheme
        musclScheme.omega = omega

        solver = Solver(mesh, Q, musclScheme)
        solver.init(init_fun)

        # CFL condition
        c = 0.5
        dx = 1 / nx
        dt = c * dx
        T = 0.1

        x = solver.mesh.cells.centroids[..., 0]
        x = x.reshape(x.size)
        Nt = int(np.ceil(T / dt))
        for t in np.linspace(0, Nt * dt, Nt + 1):
            u = solver.mesh.cells.values[..., 0]
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
        plt.ylabel(r"erreur $L^2$")
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

    assert (
        order > 2 - eps and not (musclScheme.omega == 1 / 3 * (2 * c - np.sign(c)))
    ) or (order > 3 - eps and musclScheme.omega == 1 / 3 * (2 * c - np.sign(c)))
