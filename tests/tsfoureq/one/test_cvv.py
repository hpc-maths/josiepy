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
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.tsfoureq.solver import TSFourEqSolver
from josie.tsfoureq.exact import Exact
from josie.tsfoureq.state import Q
from josie.tsfoureq.eos import TwoPhaseEOS, LinearizedGas


from josie.twofluid.fields import Phases
from tests.tsfoureq.one.conftest import RiemannState, RiemannExactProblem

eps = 1e-7
riemann_state = RiemannExactProblem(
    left=RiemannState(alphabar=1.0 - eps, ad=0, rho1=100.0, rho2=1e4, U=0.0),
    right=RiemannState(alphabar=eps, ad=0, rho1=1.0, rho2=1e3, U=0.0),
    final_time=0.03,
    xd=0.3,
    CFL=0.5,
    left_star=RiemannState(
        alphabar=1.0 - eps,
        ad=0,
        rho1=98.0806003852,
        rho2=9808.06003852,
        U=0.0581487538354,
    ),
    right_star=RiemannState(
        alphabar=eps,
        ad=0,
        rho1=1.00388410482,
        rho2=1003.88410482,
        U=0.0581487538354,
    ),
)


def relative_error(a, b):
    return np.abs(a - b)


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


def sol_exact_riemann(
    state_L: RiemannState,
    state_R: RiemannState,
    state_L_star: RiemannState,
    state_R_star: RiemannState,
    eos,
    x: np.ndarray,
    t: float,
):
    sol = np.zeros((x.size, len(Q.fields)))

    Q_L = riemann2Q(state_L, eos)
    Q_R = riemann2Q(state_R, eos)
    Q_L_star = riemann2Q(state_L_star, eos)
    Q_R_star = riemann2Q(state_R_star, eos)

    U_L = Q_L[Q.fields.U]
    c_L = Q_L[Q.fields.cFd]
    alphabar_L = Q_L[Q.fields.abar]
    ad_L = Q_L[Q.fields.ad]

    U_star = Q_L_star[Q.fields.U]
    c_L_star = Q_L_star[Q.fields.cFd]
    P_star = Q_L_star[Q.fields.pbar]

    U_R = Q_R[Q.fields.U]
    c_R = Q_R[Q.fields.cFd]
    rho_R = Q_R[Q.fields.rho]
    P_R = Q_R[Q.fields.pbar]
    ad_R = Q_R[Q.fields.ad]

    # left state
    ind = np.where(x / t < U_L - c_L)
    sol[ind, :] = Q_L
    # 1-rarefaction state
    ind = np.where((x / t > U_L - c_L) * (x / t < U_star - c_L_star))
    alphabar_fan = alphabar_L
    ad_fan = Exact.solveAlpha1dFan(
        (U_L - (x[ind] / t)) / c_L / (1 - ad_L),
        ad_L * np.ones_like(x)[ind],
    )
    U_fan = x[ind] / t + c_L * (1 - ad_L) / (1 - ad_fan)
    rho1_fan = state_L.rho1 * np.exp((U_L - U_fan) / c_L / (1 - ad_L))
    rho2_fan = state_L.rho2 * np.exp((U_L - U_fan) / c_L / (1 - ad_L))
    Q_L_fan = np.array(
        [
            np.array(
                riemann2Q(
                    RiemannState(
                        alphabar=alphabar_fan,
                        ad=ad_fan[i],
                        rho1=rho1_fan[i],
                        rho2=rho2_fan[i],
                        U=U_fan[i],
                    ),
                    eos,
                ),
            )
            for i in range(len(ad_fan))
        ]
    )
    sol[ind[0], :] = Q_L_fan.reshape((ind[0].size, len(Q.fields)))

    # left star region
    ind = np.where((x / t < U_star) * (x / t > U_star - c_L_star))
    sol[ind, :] = Q_L_star

    # right star region
    r = 1 + (1 - ad_R) / (ad_R + (rho_R * c_R**2 * (1 - ad_R)) / (P_star - P_R))
    s = U_star + (U_R - U_star) / (1 - r)
    ind = np.where((x / t > U_star) * (x / t < s))
    sol[ind, :] = Q_R_star

    # right state
    ind = np.where(x / t > s)
    sol[ind, :] = Q_R

    return sol


def test_cvv(Scheme, plot, animate, request):
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

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(50, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]

        cells.values[np.where(xc > riemann_state.xd), ...] = Q_right
        cells.values[np.where(xc <= riemann_state.xd), ...] = Q_left

    scheme = Scheme(eos, do_relaxation=True)
    solver = TSFourEqSolver(mesh, scheme)
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

        cmap = plt.get_cmap("tab10")
        (im1,) = ax1.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im2,) = ax1.plot([], [], ":", color=cmap(1), label="Numerical")
        (im3,) = ax2.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im4,) = ax2.plot([], [], ":", color=cmap(1), label="Numerical")
        (im5,) = ax3.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im6,) = ax3.plot([], [], ":", color=cmap(1), label="Numerical")
        (im7,) = ax4.plot([], [], "-k", label="Exact", linewidth=1.0)
        (im8,) = ax4.plot([], [], ":", color=cmap(1), label="Numerical")

        alphabar_data = []
        rhoU_data = []
        arho1_data = []
        ad_data = []

        x = cells.centroids[..., 0, 0, Direction.X]

        def init():
            ax1.set_xlim(0, 1)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlabel("x")
            ax1.set_ylabel(r"$\overline{\alpha}_1$")

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
            ax3.set_ylabel(r"$\alpha_1\rho_1=\overline{\alpha}_1\overline{\rho}_1$")

            # ax4.set_xlim(0, 1)
            # if riemann_state.xd == 0.25:
            #     ax4.set_ylim(-50, 1050)
            # else:
            #     ax4.set_ylim(-50, 1250)
            # ax4.set_xlabel("x")
            # ax4.set_ylabel(
            #     r"$\alpha_2\rho_2=\overline{\alpha}_2\overline{\rho}_2$"
            # )

            # ax5.set_xlim(0, 1)
            # ax5.set_ylim(-0.05, 1.05)
            # ax5.set_xlabel("x")
            # ax5.set_ylabel(r"$\alpha_1$")

            ax4.set_xlim(0, 1)
            if riemann_state.left.ad == 0.2 and riemann_state.right.ad == 0.2:
                ax4.set_ylim(0.195, 0.205)
            else:
                ax4.set_ylim(-0.05, 1.05)
            ax4.set_xlabel("x")
            ax4.set_ylabel(r"$\alpha_1^d$")

            x = cells.centroids[..., Direction.X]
            x = x.reshape(x.size)

            # Legend
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()
            # ax5.legend()
            # ax6.legend()

            return (
                im1,
                im2,
                im3,
                im4,
                im5,
                im6,
                im7,
                im8,
                # im9,
                # im10,
                # im11,
                # im12,
            )

    if plot:
        # _, _, _, _, _, _, _, _, _, _, _, _ = init()
        _, _, _, _, _, _, _, _ = init()

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
            len(alphabar_data) + 1 < t // time_interval or t == 0 or allFrames
        ):
            cells = solver.mesh.cells
            alphabar_data.append(np.array(cells.values[:, 0, 0, Q.fields.abar]))
            rhoU_data.append(np.array(cells.values[:, 0, 0, Q.fields.rhoU]))
            arho1_data.append(np.array(cells.values[:, 0, 0, Q.fields.arho1]))
            # arho2_data.append(
            #     np.array(cells.values[:, 0, 0, Q.fields.arho2])
            # )
            # alpha_data.append(
            #     np.array(cells.values[:, 0, 0, Q.fields.abar])
            #     * (1 - np.array(cells.values[:, 0, 0, Q.fields.ad]))
            # )
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

        # arho2 = cells.values[..., Q.fields.arho2]
        # arho2 = arho2.reshape(arho2.size)

        ad = cells.values[..., Q.fields.ad]
        ad = ad.reshape(ad.size)

        # alpha = cells.values[..., Q.fields.abar]
        # alpha = alpha.reshape(alpha.size) * (1 - ad)

        sol_exact = sol_exact_riemann(
            riemann_state.left,
            riemann_state.right,
            riemann_state.left_star,
            riemann_state.right_star,
            eos,
            x - riemann_state.xd,
            t,
        )
        im1.set_data(x, sol_exact[..., Q.fields.abar])
        im2.set_data(x, alphabar)
        im3.set_data(x, sol_exact[..., Q.fields.rhoU])
        im4.set_data(x, rhoU)
        im5.set_data(x, sol_exact[..., Q.fields.arho1])
        im6.set_data(x, arho1)
        # im7.set_data(x, sol_exact[..., Q.fields.arho2])
        # im8.set_data(x, arho2)
        # im9.set_data(
        #     x,
        #     sol_exact[..., Q.fields.abar]
        #     * (1 - sol_exact[..., Q.fields.ad]),
        # )
        # im10.set_data(
        #     x,
        #     alpha,
        # )
        im7.set_data(x, sol_exact[..., Q.fields.ad])
        im8.set_data(x, ad)

        plt.tight_layout()
        plt.show()
        plt.close()

    if animate:

        def update(frame):
            im1.set_data(x, alphabar_data[frame])
            im2.set_data(x, rhoU_data[frame])
            im3.set_data(x, arho1_data[frame])
            # im4.set_data(x, arho2_data[frame])
            # Warning alpha data twice
            # im5.set_data(x, alpha_data[frame])
            im6.set_data(x, ad_data[frame])
            return ax1, ax2, ax3, ax4, im1, im2, im3, im4, im5, im6
            # return ax1, ax2, ax3, ax4, ax5, ax6, im1, im2, im3, im4, im5, im6

        _ = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, len(alphabar_data)),
            init_func=init,
            blit=True,
            interval=200,
        )
        plt.show()
