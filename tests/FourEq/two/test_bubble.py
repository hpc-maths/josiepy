# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" Testing the numerical schemes on the solution provided in Toro, Eleuterio
F. Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical
Introduction. 3rd ed. Berlin Heidelberg: Springer-Verlag, 2009.
https://doi.org/10.1007/b79761, page 129 """

import matplotlib.pyplot as plt
import numpy as np


from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.FourEq.solver import FourEqSolver
from josie.FourEq.state import Q
from josie.FourEq.eos import TwoPhaseEOS, LinearizedGas

from josie.FourEq.exact import Exact
from josie.general.schemes.space import MUSCL_Hancock
from josie.general.schemes.space.limiters import MinMod

from josie.general.schemes.time.euler import ExplicitEuler
from josie.bc import make_periodic, Direction
from mpl_toolkits.axes_grid1 import make_axes_locatable


from josie.twofluid.fields import Phases


def relative_error(a, b):
    return np.abs(a - b)


class FourEqScheme(MUSCL_Hancock, MinMod, Exact, ExplicitEuler):
    pass


def test_bubble(plot, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1e3, c0=15.0),
        phase2=LinearizedGas(p0=1e5, rho0=1.0, c0=3.0),
    )

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(100, 100)
    mesh.generate()

    # Set default values for radius, pressure and volume fraction
    R = 0.25
    P0 = 1e5
    Vol = 8.0 * R * 5.0 * R
    x0 = 2.5 * R
    y0 = 4.0 * R

    eps = (np.pi * R**2) / Vol

    # Compute reference values to pass from adimensional to dimensional
    Vb = 4.0 / 3.0 * np.pi * R**3
    D = (6.0 * Vb / np.pi) ** (1.0 / 3.0)

    k = 0.038 / D  # Dimensional wave number

    cl = 1500.0
    omega = k * cl
    f = omega / (2.0 * np.pi)
    rhol = 1e3
    Pa = 11.0 * rhol * cl * D * f  # Dimensional pressure amplitude

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]
        yc = cells.centroids[..., 1]

        # Locate the bubble
        bubbleBool = (xc - x0) ** 2 + (yc - y0) ** 2 < R**2

        fields = Q.fields

        coord = np.where(bubbleBool)

        # Everywhere
        alpha = 1.0 - eps
        p = P0 + Pa * np.cos(2.0 * np.pi * k * yc)
        rho1 = eos[Phases.PHASE1].rho(p)
        rho2 = eos[Phases.PHASE2].rho(p)
        c1 = eos[Phases.PHASE1].sound_velocity(rho1)
        c2 = eos[Phases.PHASE2].sound_velocity(rho2)
        arho1 = alpha * rho1
        arho2 = (1.0 - alpha) * rho2
        rho = arho1 + arho2

        cells.values[..., fields.arho] = alpha * rho
        cells.values[..., fields.rhoU] = 0.0
        cells.values[..., fields.rhoV] = 0.0
        cells.values[..., fields.rho] = arho1 + arho2
        cells.values[..., fields.U] = 0.0
        cells.values[..., fields.V] = 0.0
        cells.values[..., fields.P] = p
        cells.values[..., fields.c] = np.sqrt((arho1 * c1**2 + arho2 * c2**2) / rho)
        cells.values[..., fields.alpha] = alpha
        cells.values[..., fields.arho1] = arho1
        cells.values[..., fields.p1] = p
        cells.values[..., fields.c1] = c1
        cells.values[..., fields.arho2] = arho2
        cells.values[..., fields.p2] = p
        cells.values[..., fields.c2] = c2

        # Bubble
        alpha = eps
        arho1 = alpha * rho1
        arho2 = (1.0 - alpha) * rho2
        rho = arho1 + arho2

        cells.values[coord + (fields.arho,)] = (alpha * rho)[coord]
        cells.values[coord + (fields.alpha,)] = alpha
        cells.values[coord + (fields.rho,)] = rho[coord]
        cells.values[coord + (fields.c,)] = np.sqrt(
            (arho1 * c1**2 + arho2 * c2**2) / rho
        )[coord]
        cells.values[coord + (fields.arho1,)] = arho1[coord]
        cells.values[coord + (fields.arho2,)] = arho2[coord]

    scheme = FourEqScheme(eos, do_relaxation=True)
    solver = FourEqSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = 0.4 / f  # Dimensional final time
    final_time = 0.10  # to have a short test
    t = 0.0
    CFL = 0.8

    cells = solver.mesh.cells

    if plot:
        # Plot initial solution

        fig = plt.figure()
        fig.suptitle(request.node.name)
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

        x = cells.centroids[..., 0, Direction.X]
        y = cells.centroids[..., 0, Direction.Y]

        alpha = cells.values[..., 0, Q.fields.alpha]

        rho = cells.values[..., 0, Q.fields.rho]

        p = cells.values[..., 0, Q.fields.P]

        p1 = cells.values[..., 0, Q.fields.p1]

        im1 = ax1.contourf(x, y, alpha)
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\alpha$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")

        im2 = ax2.contourf(x, y, rho)
        ax2.set_xlabel("x")
        ax2.set_ylabel(r"$\rho$")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax, orientation="vertical")

        im3 = ax3.contourf(x, y, p)
        ax3.set_xlabel("x")
        ax3.set_ylabel(r"$p$")
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax, orientation="vertical")

        im4 = ax4.contourf(x, y, p1)
        ax4.set_xlabel("x")
        ax4.set_ylabel(r"$p_1$")
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax, orientation="vertical")

        plt.tight_layout()
        plt.show()
        plt.close()

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
        break

    if plot:
        # Plot final step solution

        fig = plt.figure()
        fig.suptitle(request.node.name)
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

        x = cells.centroids[..., 0, Direction.X]
        y = cells.centroids[..., 0, Direction.Y]

        alpha = cells.values[..., 0, Q.fields.alpha]

        rho = cells.values[..., 0, Q.fields.rho]

        p = cells.values[..., 0, Q.fields.P]

        p1 = cells.values[..., 0, Q.fields.p1]

        im1 = ax1.contourf(x, y, alpha)
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\alpha$")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax, orientation="vertical")

        im2 = ax2.contourf(x, y, rho)
        ax2.set_xlabel("x")
        ax2.set_ylabel(r"$\rho$")
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax, orientation="vertical")

        im3 = ax3.contourf(x, y, p)
        ax3.set_xlabel("x")
        ax3.set_ylabel(r"$p$")
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, cax=cax, orientation="vertical")

        im4 = ax4.contourf(x, y, p1)
        ax4.set_xlabel("x")
        ax4.set_ylabel(r"$p_1$")
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im4, cax=cax, orientation="vertical")

        plt.tight_layout()
        plt.show()
        plt.close()
