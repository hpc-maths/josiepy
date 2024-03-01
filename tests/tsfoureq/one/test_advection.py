# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


from josie.bc import Dirichlet
from josie.boundary import Line
from josie.math import Direction
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell, MUSCLCell
from josie.mesh.cellset import MeshCellSet
from josie.general.schemes.space.muscl import MUSCL
from josie.tsfoureq.solver import TSFourEqSolver
from josie.tsfoureq.state import Q
from josie.tsfoureq.eos import TwoPhaseEOS, LinearizedGas


from josie.twofluid.fields import Phases
from tests.tsfoureq.one.conftest import RiemannState, RiemannProblem


def riemann2Q(state, eos):
    """Wrap all the operations to create a complete TSFourEq state from the
    initial Riemann Problem data
    """
    # BC
    arho1 = state.alphabar * state.rho1 * (1 - state.ad)
    arho2 = (1.0 - state.alphabar) * state.rho2 * (1 - state.ad)
    arho1d = eos[Phases.PHASE1].rho0 * state.ad
    rho = arho1 + arho2 + arho1d
    arho = state.alphabar * rho
    rhoU = rho * state.U
    rhoV = 0.0
    V = 0.0
    p1 = eos[Phases.PHASE1].p(state.rho1)
    p2 = eos[Phases.PHASE2].p(state.rho2)
    c1 = eos[Phases.PHASE1].sound_velocity(state.rho1)
    c2 = eos[Phases.PHASE2].sound_velocity(state.rho2)
    P = state.alphabar * p1 + (1.0 - state.alphabar) * p2
    c = np.sqrt((arho1 * c1**2 + arho2 * c2**2) / rho) / (1 - state.ad)

    return Q(
        arho,
        rhoU,
        rhoV,
        rho,
        state.U,
        V,
        P,
        c,
        state.alphabar,
        arho1,
        p1,
        c1,
        arho2,
        p2,
        c2,
        arho1d,
        state.ad,
    )


eps = 0.1

advectionProblem = RiemannProblem(
    left=RiemannState(alphabar=1.0 - eps, ad=0, rho1=1.0, rho2=1.0e3, U=0.15),
    right=RiemannState(alphabar=eps, ad=0, rho1=1.0, rho2=1.0e3, U=0.15),
    final_time=0.33,
    xd=0.25,
    CFL=0.5,
)


def test_advection(Scheme, plot, animate, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1.0, c0=3.0),
        phase2=LinearizedGas(p0=1e5, rho0=1e3, c0=15.0),
    )

    Q_left = riemann2Q(advectionProblem.left, eos)
    Q_right = riemann2Q(advectionProblem.right, eos)
    xd = advectionProblem.xd

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    top.bc = None
    bottom.bc = None

    if issubclass(Scheme, MUSCL):
        mesh = Mesh(left, bottom, right, top, MUSCLCell)
    else:
        mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(50, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]

        cells.values[np.where(xc > xd), ...] = Q_right
        cells.values[np.where(xc <= xd), ...] = Q_left

    scheme = Scheme(eos, do_relaxation=True)
    solver = TSFourEqSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = advectionProblem.final_time
    t = 0.0
    CFL = advectionProblem.CFL

    cells = solver.mesh.cells
    dt = scheme.CFL(cells, CFL)

    if plot or animate:
        fig = plt.figure()
        fig.suptitle(request.node.name)
        ax1 = plt.subplot(321)
        ax2 = plt.subplot(322)
        ax3 = plt.subplot(323)
        ax4 = plt.subplot(324)
        ax5 = plt.subplot(325)
        ax6 = plt.subplot(326)

        cmap = plt.get_cmap("tab10")
        (im1,) = ax1.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im2,) = ax1.plot([], [], ":", color=cmap(1), label="Numerical")
        (im3,) = ax2.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im4,) = ax2.plot([], [], ":", color=cmap(1), label="Numerical")
        (im5,) = ax3.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im6,) = ax3.plot([], [], ":", color=cmap(1), label="Numerical")
        (im7,) = ax4.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im8,) = ax4.plot([], [], ":", color=cmap(1), label="Numerical")
        (im9,) = ax5.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im10,) = ax5.plot([], [], ":", color=cmap(1), label="Numerical")
        (im11,) = ax6.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im12,) = ax6.plot([], [], ":", color=cmap(1), label="Numerical")

        alphabar_data = []
        rhoU_data = []
        arho1_data = []
        arho2_data = []
        alpha_data = []
        ad_data = []

        x = cells.centroids[..., 0, 0, Direction.X]

        def init():
            ax1.set_xlim(0, 1)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlabel("x")
            ax1.set_ylabel(r"$\overline{\alpha}_1$")

            ax2.set_xlim(0, 1)
            if advectionProblem.xd == 0.25:
                ax2.set_ylim(-5, 155)
            else:
                ax2.set_ylim(-5, 65)
            ax2.set_xlabel("x")
            ax2.set_ylabel(r"$\rho U$")

            ax3.set_xlim(0, 1)
            if advectionProblem.xd == 0.25:
                ax3.set_ylim(-0.05, 1.05)
            else:
                ax3.set_ylim(-5, 105)
            ax3.set_xlabel("x")
            ax3.set_ylabel(r"$\alpha_1\rho_1=\overline{\alpha}_1\overline{\rho}_1$")

            ax4.set_xlim(0, 1)
            if advectionProblem.xd == 0.25:
                ax4.set_ylim(-50, 1050)
            else:
                ax4.set_ylim(-50, 1250)
            ax4.set_xlabel("x")
            ax4.set_ylabel(r"$\alpha_2\rho_2=\overline{\alpha}_2\overline{\rho}_2$")

            ax5.set_xlim(0, 1)
            ax5.set_ylim(-0.05, 1.05)
            ax5.set_xlabel("x")
            ax5.set_ylabel(r"$\alpha_1$")

            ax6.set_xlim(0, 1)
            if advectionProblem.left.ad == 0.2:
                ax6.set_ylim(0.195, 0.205)
            else:
                ax6.set_ylim(-0.05, 1.05)
            ax6.set_xlabel("x")
            ax6.set_ylabel(r"$\alpha_1^d$")

            x = cells.centroids[..., Direction.X]
            x = x.reshape(x.size)

            # Legend
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()
            ax5.legend()
            ax6.legend()

            return (
                im1,
                im2,
                im3,
                im4,
                im5,
                im6,
                im7,
                im8,
                im9,
                im10,
                im11,
                im12,
            )

    if plot:
        _, _, _, _, _, _, _, _, _, _, _, _ = init()

    if animate:
        nFrames = 30
        allFrames = True
        time_interval = advectionProblem.final_time / nFrames

    # TODO: Use josie.io.strategy and josie.io.writer to save the plot every
    # time instant.  In particular it might useful to choose a Strategy (or
    # multiple strategies) and append to each strategy some "executors" that do
    # stuff with the Solver data
    # final_time = 4 * dt
    while t <= final_time:
        if animate and (
            len(alpha_data) + 1 < t // time_interval or t == 0 or allFrames
        ):
            cells = solver.mesh.cells
            alphabar_data.append(np.array(cells.values[:, 0, 0, Q.fields.abar]))
            rhoU_data.append(np.array(cells.values[:, 0, 0, Q.fields.rhoU]))
            arho1_data.append(np.array(cells.values[:, 0, 0, Q.fields.arho1]))
            arho2_data.append(np.array(cells.values[:, 0, 0, Q.fields.arho2]))
            alpha_data.append(
                np.array(cells.values[:, 0, 0, Q.fields.abar])
                * (1 - np.array(cells.values[:, 0, 0, Q.fields.ad]))
            )
            ad_data.append(np.array(cells.values[:, 0, 0, Q.fields.ad]))
        dt = scheme.CFL(cells, CFL)
        # TODO: Basic check. The best would be to check against analytical
        # solution
        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    # Check that we reached the final time
    assert t >= final_time

    if plot:
        # Plot final step solution

        alphabar = cells.values[..., Q.fields.abar]
        alphabar = alphabar.reshape(alphabar.size)

        rhoU = cells.values[..., Q.fields.rhoU]
        rhoU = rhoU.reshape(rhoU.size)

        arho1 = cells.values[..., Q.fields.arho1]
        arho1 = arho1.reshape(arho1.size)

        arho2 = cells.values[..., Q.fields.arho2]
        arho2 = arho2.reshape(arho2.size)

        ad = cells.values[..., Q.fields.ad]
        ad = ad.reshape(ad.size)

        alpha = cells.values[..., Q.fields.abar]
        alpha = alpha.reshape(alpha.size) * (1 - ad)

        U = Q_left[Q.fields.U]

        sol_exact = np.einsum("i,j->ji", Q_left, (x - xd) / t < U) + np.einsum(
            "i,j->ji", Q_right, (x - xd) / t >= U
        )

        im1.set_data(x, sol_exact[..., Q.fields.abar])
        im2.set_data(x, alphabar)
        im3.set_data(x, sol_exact[..., Q.fields.rhoU])
        im4.set_data(x, rhoU)
        im5.set_data(x, sol_exact[..., Q.fields.arho1])
        im6.set_data(x, arho1)
        im7.set_data(x, sol_exact[..., Q.fields.arho2])
        im8.set_data(x, arho2)
        im9.set_data(
            x,
            sol_exact[..., Q.fields.abar] * (1 - sol_exact[..., Q.fields.ad]),
        )
        im10.set_data(
            x,
            alpha,
        )
        im11.set_data(x, sol_exact[..., Q.fields.ad])
        im12.set_data(x, ad)

        plt.tight_layout()
        plt.show()
        plt.close()

    if animate:

        def update(frame):
            im1.set_data(x, alpha_data[frame])
            im2.set_data(x, rhoU_data[frame])
            im3.set_data(x, arho1_data[frame])
            im4.set_data(x, arho2_data[frame])
            im5.set_data(x, alpha_data[frame])
            im6.set_data(x, ad_data[frame])
            return ax1, ax2, ax3, ax4, ax5, ax6, im1, im2, im3, im4, im5, im6

        _ = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, len(alpha_data)),
            init_func=init,
            blit=True,
            interval=200,
        )
        plt.show()
