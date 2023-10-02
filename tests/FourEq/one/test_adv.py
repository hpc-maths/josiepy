# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


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
from .conftest import RiemannState


class CVVScheme(MinMod, MUSCL, Rusanov, RK2_relax):
    pass


def test_cvv(riemann2Q, plot, animate, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1.0, c0=3.0),
        phase2=LinearizedGas(p0=1e5, rho0=1e3, c0=3.0),
    )

    Q_in = riemann2Q(RiemannState(alpha=0, rho1=1.0, rho2=1.0e3, U=1.0), eos)
    Q_out = riemann2Q(RiemannState(alpha=1.0, rho1=1.0, rho2=1.0e3, U=1.0), eos)

    left, right = make_periodic(left, right, Direction.X)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    mesh.interpolate(500, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]

        center = 0.5
        width = 0.5
        cells.values[np.where(np.abs(xc - center) <= width / 2), ...] = Q_in
        cells.values[np.where(np.abs(xc - center) > width / 2), ...] = Q_out

    scheme = CVVScheme(eos, do_relaxation=True)
    solver = FourEqSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = 1
    t = 0.0
    CFL = 0.8

    cells = solver.mesh.cells
    dt = scheme.CFL(cells, CFL)

    if plot or animate:
        fig = plt.figure()
        fig.suptitle(request.node.name)
        ax1 = plt.subplot(111)

        (im1,) = ax1.plot([], [], "-", label="Numerical")

        alpha_data = []

        x = cells.centroids[..., 0, 0, Direction.X]

        def init():
            ax1.set_xlim(0, 1)
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xlabel("x")
            ax1.set_ylabel(r"$\alpha$")

            x = cells.centroids[..., Direction.X]
            x = x.reshape(x.size)

            # Legend
            ax1.legend()

            return im1

    if plot:
        _ = init()

    if animate:
        nFrames = 30
        allFrames = True
        time_interval = final_time / nFrames

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

        plt.tight_layout()
        plt.show()
        plt.close()

    if animate:

        def update(frame):
            im1.set_data(x, alpha_data[frame])
            return ax1, im1

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
            return ax1, im1

        _ = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, len(alpha_data)),
            init_func=init,
            blit=True,
            interval=200,
        )
        plt.show()
