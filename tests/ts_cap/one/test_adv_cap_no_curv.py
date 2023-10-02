# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np


from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import MUSCLCell
from josie.mesh.cellset import MeshCellSet
from josie.ts_cap.solver import TsCapSolver
from josie.ts_cap.state import Q
from josie.bn.eos import TwoPhaseEOS
from josie.FourEq.eos import LinearizedGas

from josie.ts_cap.schemes import Rusanov
from josie.general.schemes.space import MUSCL
from josie.general.schemes.space.limiters import MinMod

from josie.general.schemes.time.rk import RK2_relax
from josie.bc import make_periodic, Direction


from josie.twofluid.fields import Phases


class TsCapScheme(Rusanov, RK2_relax, MUSCL, MinMod):
    pass


def test_dummy(plot, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1e0, c0=1e1),
        phase2=LinearizedGas(p0=1e5, rho0=1e0, c0=1e1),
    )

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    mesh.interpolate(60, 10)
    mesh.generate()

    # sigma = 7.3e-3
    sigma = 1e-2
    # sigma = 0
    Hmax = 1e3
    dx = mesh.cells._centroids[1, 1, 0, 0] - mesh.cells._centroids[0, 1, 0, 0]
    dy = mesh.cells._centroids[1, 1, 0, 1] - mesh.cells._centroids[1, 0, 0, 1]
    norm_grada_min = 1e-3

    scheme = TsCapScheme(
        eos,
        sigma,
        Hmax,
        dx,
        dy,
        norm_grada_min,
    )

    def toState(
        cells: MeshCellSet,
        index: tuple,
        abar: float,
        ad: float,
        U: float,
        V: float,
    ):
        fields = Q.fields

        arho1 = eos[Phases.PHASE1].rho0 * abar * (1 - ad)
        arho2 = eos[Phases.PHASE2].rho0 * (1 - abar) * (1 - ad)
        arho1d = eos[Phases.PHASE1].rho0 * ad
        capSigma = 0
        rho = arho1 + arho2 + arho1d
        rhoU = rho * U
        rhoV = rho * V

        cells.values[index + (fields.arho1,)] = arho1
        cells.values[index + (fields.arho1d,)] = arho1d
        cells.values[index + (fields.arho2,)] = arho2
        cells.values[index + (fields.ad,)] = ad
        cells.values[index + (fields.capSigma,)] = capSigma
        cells.values[index + (fields.rhoU,)] = rhoU
        cells.values[index + (fields.rhoV,)] = rhoV
        cells.values[index + (fields.abarrho,)] = rho * abar

    def init_fun(cells: MeshCellSet):
        x_c = cells.centroids[..., 0]
        eps = 1e-4
        x_left = np.where((x_c < 0.75 + eps) * (x_c >= 0.25 + eps))
        x_right = np.where(1 - ((x_c < 0.75 + eps) * (x_c >= 0.25 + eps)))

        abar_L = 1
        abar_R = 0
        ad = 0
        U = 1
        V = 0

        toState(cells, x_left, abar_L, ad, U, V)
        toState(cells, x_right, abar_R, ad, U, V)

    solver = TsCapSolver(mesh, scheme)
    solver.init(init_fun)
    solver.mesh.update_ghosts(0)
    solver.scheme.auxilliaryVariableUpdate(solver.mesh.cells._values[..., 0, :])
    solver.mesh.update_ghosts(0)
    # print(solver.mesh.cells._values[..., 0, Q.fields.n_y])
    # exit()

    final_time = 1
    t = 0.0
    CFL = 0.8

    cells = solver.mesh.cells

    if plot:
        # Plot initial solution

        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        # ax3 = plt.subplot(223)
        # ax4 = plt.subplot(224)

        # x = cells._centroids[..., 0, Direction.X]
        # y = cells._centroids[..., 0, Direction.Y]

        abar = cells._values[..., 0, Q.fields.abar]

        H = cells._values[..., 0, Q.fields.V]

        #     p = cells.values[..., 0, Q.fields.P]

        #     p1 = cells.values[..., 0, Q.fields.p1]

        _ = ax1.imshow(abar)
        #     ax1.set_xlabel("x")
        #     ax1.set_ylabel(r"$\alpha$")
        #     divider = make_axes_locatable(ax1)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im1, cax=cax, orientation="vertical")

        _ = ax2.imshow(H)
    #     ax2.set_xlabel("x")
    #     ax2.set_ylabel(r"$\rho$")
    #     divider = make_axes_locatable(ax2)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(im2, cax=cax, orientation="vertical")

    #     im3 = ax3.contourf(x, y, p)
    #     ax3.set_xlabel("x")
    #     ax3.set_ylabel(r"$p$")
    #     divider = make_axes_locatable(ax3)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(im3, cax=cax, orientation="vertical")

    #     im4 = ax4.contourf(x, y, p1)
    #     ax4.set_xlabel("x")
    #     ax4.set_ylabel(r"$p_1$")
    #     divider = make_axes_locatable(ax4)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     fig.colorbar(im4, cax=cax, orientation="vertical")

    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()

    # TODO: Use josie.io.strategy and josie.io.writer to save the plot every
    # time instant.  In particular it might useful to choose a Strategy (or
    # multiple strategies) and append to each strategy some "executors" that do
    # stuff with the Solver data
    k = 0
    while t <= final_time:
        dt = scheme.CFL(cells, CFL)

        assert ~np.isnan(dt)
        solver.step(dt)
        # break

        t += dt
        print(f"Time: {t}, dt: {dt}")
        k += 1

    if plot:
        # Plot final step solution

        _ = plt.figure()
        # fig.suptitle(request.node.name)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        # ax3 = plt.subplot(223)
        # ax4 = plt.subplot(224)

        # x = cells._centroids[..., 0, Direction.X]
        # y = cells._centroids[..., 0, Direction.Y]

        #     p = cells.values[..., 0, Q.fields.P]

        #     p1 = cells.values[..., 0, Q.fields.p1]

        _ = ax1.imshow(cells._values[..., 0, Q.fields.abar])
        #     ax1.set_xlabel("x")
        #     ax1.set_ylabel(r"$\alpha$")
        #     divider = make_axes_locatable(ax1)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im1, cax=cax, orientation="vertical")

        _ = ax2.imshow(cells._values[..., 0, Q.fields.norm_grada])
        #     ax2.set_xlabel("x")
        #     ax2.set_ylabel(r"$\rho$")
        #     divider = make_axes_locatable(ax2)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im2, cax=cax, orientation="vertical")

        #     im3 = ax3.contourf(x, y, p)
        #     ax3.set_xlabel("x")
        #     ax3.set_ylabel(r"$p$")
        #     divider = make_axes_locatable(ax3)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im3, cax=cax, orientation="vertical")

        #     im4 = ax4.contourf(x, y, p1)
        #     ax4.set_xlabel("x")
        #     ax4.set_ylabel(r"$p_1$")
        #     divider = make_axes_locatable(ax4)
        #     cax = divider.append_axes("right", size="5%", pad=0.05)
        #     fig.colorbar(im4, cax=cax, orientation="vertical")

        #     plt.tight_layout()
        plt.show()
        plt.close()
