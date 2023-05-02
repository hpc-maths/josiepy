# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" Testing the additional limiter for conservative components of the
Euler equations provided in Berthon, Christophe. « Why the MUSCL–Hancock
Scheme Is L1-Stable ». Numerische Mathematik, nᵒ 104 (2006): 27‑46.
https://doi.org/10.1007/s00211-006-0007-4. """

import matplotlib.pyplot as plt
import numpy as np
import pytest

from josie.general.schemes.time import ExplicitEuler
from josie.general.schemes.space.muscl import MUSCL_Hancock
from josie.general.schemes.space.limiters import MinMod

from josie.bc import Dirichlet
from josie.boundary import Line
from josie.math import Direction
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.euler.eos import PerfectGas
from josie.euler.schemes import BerthonScheme, HLLC
from josie.euler.exact import Exact
from josie.euler.solver import EulerSolver
from josie.euler.state import EulerState

from tests.euler.conftest import RiemannProblem, RiemannState, RiemannSolution


def relative_error(a, b):
    return np.abs(a - b)


@pytest.fixture(params=[None, BerthonScheme])
def BScheme(request):
    yield request.param


@pytest.fixture
def Scheme(BScheme):
    """Create all the different schemes"""
    if BScheme is None:

        class ToroScheme(MUSCL_Hancock, MinMod, ExplicitEuler, HLLC):
            pass

    else:

        class ToroScheme(MUSCL_Hancock, MinMod, ExplicitEuler, HLLC, BScheme):
            pass

    return ToroScheme


riemann_states = [
    RiemannProblem(
        left=RiemannState(rho=1.0, U=0, V=0, p=1.0),
        right=RiemannState(rho=0.125, U=0, V=0, p=0.1),
        final_time=0.25,
        CFL=0.5,
        solution=RiemannSolution(
            p_star=0.30313,
            U_star=0.92745,
            rho_star_L=0.42632,
            rho_star_R=0.26557,
        ),
    ),
    RiemannProblem(
        left=RiemannState(rho=1.0, U=-2.0, V=0, p=0.4),
        right=RiemannState(rho=1.0, U=2.0, V=0, p=0.4),
        final_time=0.15,
        CFL=0.5,
        solution=RiemannSolution(
            p_star=0.00189,
            U_star=0.0,
            rho_star_L=0.02185,
            rho_star_R=0.02185,
        ),
    ),
    RiemannProblem(
        left=RiemannState(rho=1.0, U=0, V=0, p=1000),
        right=RiemannState(rho=1.0, U=0, V=0, p=0.01),
        final_time=0.012,
        CFL=0.45,
        solution=RiemannSolution(
            p_star=460.894,
            U_star=19.5975,
            rho_star_L=0.57506,
            rho_star_R=5.99924,
        ),
    ),
    RiemannProblem(
        left=RiemannState(rho=1.0, U=0, V=0, p=0.01),
        right=RiemannState(rho=1.0, U=0, V=0, p=100),
        final_time=0.035,
        CFL=0.45,
        solution=RiemannSolution(
            p_star=46.0950,
            U_star=-6.19633,
            rho_star_L=5.9924,
            rho_star_R=0.57511,
        ),
    ),
]


def riemann2Q(state, eos):
    """Wrap all the operations to create a complete Euler state from the
    initial Rieman Problem data
    """
    # BC
    rho = state.rho
    U = state.U
    V = state.V
    p = state.p
    rhoe = eos.rhoe(rho, p)
    E = rhoe / rho + 0.5 * (U**2 + V**2)
    c = eos.sound_velocity(rho, p)

    return EulerState(rho, rho * U, rho * V, rho * E, rhoe, U, V, p, c, rhoe / rho)


@pytest.mark.parametrize("riemann", sorted(riemann_states, key=id))
def test_toro(riemann, Scheme, plot, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    Q_left = riemann2Q(riemann.left, eos)
    Q_right = riemann2Q(riemann.right, eos)

    # Create exact Riemann solver
    riemann_solver = Exact(eos, Q_left, Q_right)

    # Solve the Riemann problem
    riemann_solver.solve()

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(100, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]

        cells.values[np.where(xc > 0.5), ...] = Q_right
        cells.values[np.where(xc <= 0.5), ...] = Q_left

    scheme = Scheme(eos)
    solver = EulerSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = riemann.final_time
    t = 0
    CFL = riemann.CFL

    cells = solver.mesh.cells

    if plot:
        fig = plt.figure()
        fig.suptitle(request.node.name)
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

    # TODO: Use josie.io.strategy and josie.io.writer to save the plot every
    # time instant.  In particular it might useful to choose a Strategy (or
    # multiple strategies) and append to each strategy some "executors" that do
    # stuff with the Solver data
    while t <= final_time:
        dt = scheme.CFL(cells, CFL)

        # TODO: Basic check. The best would be to check against analytical
        # solution
        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt

        print(f"Time: {t}, dt: {dt}")

    # Check that we reached the final time
    assert t >= final_time

    # Assert convergence on the final step
    x = cells.centroids[..., Direction.X]
    x = x.reshape(x.size)

    rho = cells.values[..., EulerState.fields.rho]
    rho = rho.reshape(rho.size)

    U = cells.values[..., EulerState.fields.U]
    U = U.reshape(U.size)

    p = cells.values[..., EulerState.fields.p]
    p = p.reshape(p.size)

    err_p = 0
    ref_p = 0
    err_U = 0
    ref_U = 0
    err_rho = 0
    ref_rho = 0

    for i, x_step in enumerate(x):
        R = riemann_solver.sample(x_step, t)
        err_p += (R[..., R.fields.p] - p[i]) ** 2
        ref_p += R[..., R.fields.p] ** 2
        err_U += (R[..., R.fields.U] - U[i]) ** 2
        ref_U += R[..., R.fields.U] ** 2
        err_rho += (R[..., R.fields.rho] - rho[i]) ** 2
        ref_rho += R[..., R.fields.rho] ** 2

    if plot:
        (im1,) = ax1.plot(x, rho, "-", label="Numerical")
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\rho$")

        (im2,) = ax2.plot(x, U, "-", label="Numerical")
        ax2.set_xlabel("x")
        ax2.set_ylabel("U")

        (im3,) = ax3.plot(x, p, "-", label="Numerical")
        ax3.set_xlabel("x")
        ax3.set_ylabel("p")

        # Plot the exact solution over the final step solution
        p = []
        U = []
        rho = []
        for x_step in x:
            R = riemann_solver.sample(x_step, t)
            p.append(R[..., R.fields.p])
            U.append(R[..., R.fields.U])
            rho.append(R[..., R.fields.rho])

        (im1,) = ax1.plot(x, rho, "--", label="Exact")
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\rho$")

        (im2,) = ax2.plot(x, U, "--", label="Exact")
        ax2.set_xlabel("x")
        ax2.set_ylabel("U")

        (im3,) = ax3.plot(x, p, "--", label="Exact")
        ax3.set_xlabel("x")
        ax3.set_ylabel("p")

        # Legend
        ax1.legend()
        ax2.legend()
        ax3.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

    assert err_rho / ref_rho < 0.3
    assert err_U / ref_U < 0.3
    assert err_p / ref_p < 0.3
