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
from josie.mesh.cell import MUSCLCell
from josie.mesh.cellset import MeshCellSet
from josie.FourEq.solver import FourEqSolver
from josie.FourEq.state import Q
from josie.FourEq.eos import TwoPhaseEOS, LinearizedGas


from josie.twofluid.fields import Phases


def relative_error(a, b):
    return np.abs(a - b)


def riemann2Q(state, eos):
    """Wrap all the operations to create a complete FourEq state from the
    initial Riemann Problem data
    """
    # BC
    arho1 = state.alpha * state.rho1
    arho2 = (1.0 - state.alpha) * state.rho2
    rho = arho1 + arho2
    arho = state.alpha * rho
    rhoU = rho * state.U
    rhoV = 0.0
    V = 0.0
    p1 = eos[Phases.PHASE1].p(state.rho1)
    p2 = eos[Phases.PHASE2].p(state.rho2)
    c1 = eos[Phases.PHASE1].sound_velocity(state.rho1)
    c2 = eos[Phases.PHASE2].sound_velocity(state.rho2)
    P = state.alpha * p1 + (1.0 - state.alpha) * p2
    c = np.sqrt((arho1 * c1**2 + arho2 * c2**2) / rho)

    return Q(
        arho,
        rhoU,
        rhoV,
        rho,
        state.U,
        V,
        P,
        c,
        state.alpha,
        arho1,
        p1,
        c1,
        arho2,
        p2,
        c2,
    )


def test_cvv(riemann_state, Scheme, plot, animate, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1.0, c0=3.0),
        phase2=LinearizedGas(p0=1e5, rho0=1e3, c0=15.0),
    )

    Q_left = riemann2Q(riemann_state.left, eos)
    Q_right = riemann2Q(riemann_state.right, eos)

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    mesh.interpolate(50, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]

        cells.values[np.where(xc > riemann_state.xd), ...] = Q_right
        cells.values[np.where(xc <= riemann_state.xd), ...] = Q_left

    scheme = Scheme(eos, do_relaxation=True)
    solver = FourEqSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = riemann_state.final_time
    t = 0.0
    CFL = riemann_state.CFL

    cells = solver.mesh.cells
    dt = scheme.CFL(cells, CFL)

    if plot or animate:
        fig = plt.figure()
        fig.suptitle(request.node.name)
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

        (im1,) = ax1.plot([], [], "-", label="Numerical")
        (im2,) = ax2.plot([], [], "-", label="Numerical")
        (im3,) = ax3.plot([], [], "-", label="Numerical")
        (im4,) = ax4.plot([], [], "-", label="Numerical")

        alpha_data = []
        rhoU_data = []
        arho1_data = []
        arho2_data = []

        x = cells.centroids[..., 0, 0, Direction.X]

        def init():
            ax1.set_xlim(0, 1)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlabel("x")
            ax1.set_ylabel(r"$\alpha$")

            ax2.set_xlim(0, 1)
            if riemann_state.xd == 0.25:
                ax2.set_ylim(-5, 155)
            else:
                ax2.set_ylim(-5, 65)
            ax2.set_xlabel("x")
            ax2.set_ylabel(r"$\rho U$")

            ax3.set_xlim(0, 1)
            if riemann_state.xd == 0.25:
                ax3.set_ylim(-0.05, 1.05)
            else:
                ax3.set_ylim(-5, 105)
            ax3.set_xlabel("x")
            ax3.set_ylabel(r"$\alpha_1\rho_1$")

            ax4.set_xlim(0, 1)
            if riemann_state.xd == 0.25:
                ax4.set_ylim(-50, 1050)
            else:
                ax4.set_ylim(-50, 1250)
            ax4.set_xlabel("x")
            ax4.set_ylabel(r"$\alpha_2\rho_2$")

            x = cells.centroids[..., Direction.X]
            x = x.reshape(x.size)

            # Legend
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()

            return im1, im2, im3, im4

    if plot:
        _, _, _, _ = init()

    if animate:
        nFrames = 30
        allFrames = True
        time_interval = riemann_state.final_time / nFrames

    # TODO: Use josie.io.strategy and josie.io.writer to save the plot every
    # time instant.  In particular it might useful to choose a Strategy (or
    # multiple strategies) and append to each strategy some "executors" that do
    # stuff with the Solver data
    while t <= final_time:
        if animate and (
            len(alpha_data) + 1 < t // time_interval or t == 0 or allFrames
        ):
            cells = solver.mesh.cells
            alpha_data.append(np.array(cells.values[:, 0, 0, Q.fields.alpha]))
            rhoU_data.append(np.array(cells.values[:, 0, 0, Q.fields.rhoU]))
            arho1_data.append(np.array(cells.values[:, 0, 0, Q.fields.arho1]))
            arho2_data.append(np.array(cells.values[:, 0, 0, Q.fields.arho2]))
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

        alpha = cells.values[..., Q.fields.alpha]
        alpha = alpha.reshape(alpha.size)

        rhoU = cells.values[..., Q.fields.rhoU]
        rhoU = rhoU.reshape(rhoU.size)

        arho1 = cells.values[..., Q.fields.arho1]
        arho1 = arho1.reshape(arho1.size)

        arho2 = cells.values[..., Q.fields.arho2]
        arho2 = arho2.reshape(arho2.size)

        im1.set_data(x, alpha)
        im2.set_data(x, rhoU)
        im3.set_data(x, arho1)
        im4.set_data(x, arho2)

        plt.tight_layout()
        plt.show()
        plt.close()

    if animate:

        def update(frame):
            im1.set_data(x, alpha_data[frame])
            im2.set_data(x, rhoU_data[frame])
            im3.set_data(x, arho1_data[frame])
            im4.set_data(x, arho2_data[frame])
            return ax1, ax2, ax3, ax4, im1, im2, im3, im4

        _ = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, len(alpha_data)),
            init_func=init,
            blit=True,
            interval=200,
        )
        plt.show()

    if animate:

        def update(frame):
            im1.set_data(x, alpha_data[frame])
            im2.set_data(x, rhoU_data[frame])
            im3.set_data(x, arho1_data[frame])
            im4.set_data(x, arho2_data[frame])
            return ax1, ax2, ax3, ax4, im1, im2, im3, im4

        _ = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, len(alpha_data)),
            init_func=init,
            blit=True,
            interval=200,
        )
        plt.show()
