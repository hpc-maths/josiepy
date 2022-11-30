# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.animation import ArtistAnimation


import josie.general.schemes.time as time_schemes
from josie.general.schemes.space.limiters import MUSCL_Hancock_no_limiter
from josie.dimension import MAX_DIMENSIONALITY
from josie.bc import Dirichlet
from josie.mesh.cellset import MeshCellSet
from josie.fluid.state import ConsState
from josie.solver import Solver
from josie.problem import Problem
from josie.state import SubsetState, State
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
    # u = (x > 0.4) * (x < 0.6)
    # u = 0.5 * (1 + erf(20 * (x - 0.5)))
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
    params=[member[1] for member in inspect.getmembers(time_schemes, inspect.isclass)],
)
def scheme(request):
    class Upwind(MUSCL_Hancock_no_limiter, request.param):
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


@pytest.fixture
def solver(scheme):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left.bc = Dirichlet(AdvectionConsState(0))
    right.bc = Dirichlet(AdvectionConsState(0))
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(40, 1)
    mesh.generate()
    solver = Solver(mesh, Q, scheme)
    solver.init(init_fun)

    yield solver


def test_against_real_1D(solver, plot, tol):
    """Testing against the real 1D solver"""

    nx = solver.mesh.num_cells_x

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ims = []

    # CFL condition
    c = 0.7
    dx = 1 / nx
    dt = c * dx
    T = 0.1

    x = solver.mesh.cells.centroids[..., 0]
    x = x.reshape(x.size)
    Nt = int(np.ceil(T / dt))
    dt = T / Nt

    for t in np.linspace(0, T, Nt + 1):
        u = solver.mesh.cells.values[..., 0]
        u = u.reshape(u.size)

        err = u - init(x - t)

        if plot:
            (im1,) = ax1.plot(x, u, "ro-")
            (im2,) = ax1.plot(x, init(x - t), "ks-")
            ims.append([im1, im2])
            (im_err,) = ax2.plot(x, err, "ks-")
            ims.append([im1, im2, im_err])

        # Check same solution with 1D-only
        # assert np.sum(err < tol) == len(x)
        solver.step(dt)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=100)
        plt.show()
