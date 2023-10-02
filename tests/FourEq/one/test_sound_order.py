# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from josie.twofluid.fields import Phases
from dataclasses import dataclass


from josie.boundary import Line
from josie.math import Direction
from josie.mesh import Mesh
from josie.mesh.cell import MUSCLCell
from josie.mesh.cellset import MeshCellSet
from josie.FourEq.solver import FourEqSolver
from josie.FourEq.state import Q
from josie.FourEq.eos import TwoPhaseEOS, LinearizedGas
from josie.FourEq.schemes import Rusanov

from josie.general.schemes.space.muscl import MUSCL
from josie.general.schemes.space.limiters import MinMod
from josie.general.schemes.time.rk import RK2_relax

from josie.bc import make_periodic


class CVVScheme(MinMod, MUSCL, Rusanov, RK2_relax):
    pass


def pressure2Q(state, eos):
    """Wrap all the operations to create a complete FourEq state from the
    initial Riemann Problem data
    """
    # BC
    rho1 = eos[Phases.PHASE1].rho(state.P)
    rho2 = eos[Phases.PHASE2].rho(state.P)

    arho1 = state.alpha * rho1
    arho2 = (1.0 - state.alpha) * rho2
    rho = arho1 + arho2

    arho = state.alpha * rho
    rhoU = rho * state.U
    rhoV = 0.0
    V = 0.0

    if state.alpha > 0.0:
        p1 = state.P
    else:
        p1 = np.nan
    if state.alpha < 1.0:
        p2 = state.P
    else:
        p2 = np.nan
    c1 = eos[Phases.PHASE1].sound_velocity(rho1)
    c2 = eos[Phases.PHASE2].sound_velocity(rho2)
    c = np.sqrt((arho1 * c1**2 + arho2 * c2**2) / rho)

    return Q(
        arho,
        rhoU,
        rhoV,
        rho,
        state.U,
        V,
        state.P,
        c,
        state.alpha,
        arho1,
        p1,
        c1,
        arho2,
        p2,
        c2,
    )


@dataclass
class InitState:
    alpha: float
    P: float
    U: float


def test_cvv(plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1.0, c0=340),
        phase2=LinearizedGas(p0=1e5, rho0=1e3, c0=370),
    )

    left, right = make_periodic(left, right, Direction.X)
    top.bc = None
    bottom.bc = None
    L2_err = []
    ref_sol_nx = 6000
    nx_tab = [50, 100, 300, 500, 1000, 2000, 3000]
    final_time = 1e-3
    plt.figure()

    def init_P(x):
        P0 = 101325
        x0 = 0.5
        dP = 1
        return P0 + dP * np.sinc(4 * (x - x0)) * (np.abs(x - x0) <= 1 / 4) ** 4

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0, 0, 0]

        for i, xc in enumerate(xc):
            cells.values[i, 0, 0, :] = pressure2Q(
                InitState(
                    alpha=0.12,
                    P=init_P(xc),
                    U=0.0,
                ),
                eos,
            )

    scheme = CVVScheme(eos, do_relaxation=True)

    # compute ref
    print("Computing ref...")
    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    mesh.interpolate(ref_sol_nx, 1)
    mesh.generate()
    solver = FourEqSolver(mesh, scheme)
    solver.init(init_fun)
    dt = scheme.CFL(solver.mesh.cells, 0.8)

    Nt = int(np.ceil(final_time / dt))
    for t in np.linspace(0, Nt * dt, Nt + 1):
        Qt = solver.mesh.cells.values
        solver.step(dt)
        print(f"Time: {t}, dt: {dt}", end="\r")
    P_sol = solver.mesh.cells.values[:, 0, 0, Q.fields.P]
    xc_sol = mesh.cells.centroids[:, 0, 0, 0]
    sorter = np.argsort(xc_sol)

    for nx in nx_tab:
        print(nx)
        mesh = Mesh(left, bottom, right, top, MUSCLCell)
        mesh.interpolate(nx, 1)
        mesh.generate()
        solver = FourEqSolver(mesh, scheme)
        solver.init(init_fun)
        dt = scheme.CFL(solver.mesh.cells, 0.8)

        Nt = int(np.ceil(final_time / dt))
        for t in np.linspace(0, Nt * dt, Nt + 1):
            Qt = solver.mesh.cells.values
            solver.step(dt)
            print(f"Time: {t}, dt: {dt}", end="\r")

        # if plot:
        #     plt.plot(solver.mesh.cells.centroids[:, 0, 0, 0], Qt[:, 0, 0, Q.fields.P])
        #     plt.show()

        ind = sorter[
            np.searchsorted(
                xc_sol, solver.mesh.cells.centroids[:, 0, 0, 0], sorter=sorter
            )
        ]
        err = Qt[..., 0, 0, Q.fields.P] - P_sol[ind]

        L2_err.append(np.linalg.norm(err[..., Q.fields.P]) * np.sqrt(1 / nx))
        if np.isnan(L2_err[-1]):
            exit()

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
        plt.title(r"L2 error")
        plt.legend(loc="lower left")

        plt.show()

    eps = 0.2
    order = -np.linalg.lstsq(
        np.vstack([np.log(nx_tab), np.ones(len(nx_tab))]).T,
        np.log(L2_err),
        rcond=None,
    )[0][0]
    print(order)

    assert order > 2 - eps
