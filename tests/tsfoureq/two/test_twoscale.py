# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


from josie.bc import Dirichlet, Direction, make_periodic
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.tsfoureq.solver import TSFourEqSolver
from josie.tsfoureq.state import Q
from josie.tsfoureq.eos import TwoPhaseEOS, LinearizedGas


from josie.twofluid.fields import Phases


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


def test_toro(riemann_state, Scheme, plot, animate, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [2, 0])
    right = Line([2, 0], [2, 1])
    top = Line([0, 1], [2, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1.0, c0=3.0),
        phase2=LinearizedGas(p0=1e5, rho0=1e3, c0=15.0),
    )

    Q_left = riemann2Q(riemann_state.left, eos)
    Q_right = riemann2Q(riemann_state.right, eos)
    Q_small_scale = riemann2Q(riemann_state.small_scale, eos)

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(200, 100)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]
        yc = cells.centroids[..., 1]
        x0 = 1
        y0 = 0.5
        # R = 0.1

        idx_left = np.where(xc <= riemann_state.xd)
        idx_right = np.where(xc > riemann_state.xd)
        # Circle
        # idx_small_scale = np.where((xc - x0) ** 2 + (yc - y0) ** 2 < R ** 2)
        # central beam
        idx_small_scale = np.where((np.abs(yc - y0) < 0.1) * (np.abs(xc - x0) < 0.5))

        cells.values[idx_left[0], idx_left[1], :] = Q_left
        cells.values[idx_right[0], idx_right[1], :] = Q_right
        cells.values[idx_small_scale[0], idx_small_scale[1], :] = Q_small_scale

    scheme = Scheme(eos, do_relaxation=True)
    solver = TSFourEqSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = riemann_state.final_time
    t = 0.0
    CFL = riemann_state.CFL

    cells = solver.mesh.cells
    dt = scheme.CFL(cells, CFL)

    def init():
        return ax, im

    if plot or animate:
        fig, ax = plt.subplots()

        im = ax.imshow([[]], origin="lower", extent=(0, 2, 0, 1), aspect="equal")

        data = []

    if animate:
        nFrames = 30
        allFrames = False
        time_interval = riemann_state.final_time / nFrames

    # TODO: Use josie.io.strategy and josie.io.writer to save the plot every
    # time instant.  In particular it might useful to choose a Strategy (or
    # multiple strategies) and append to each strategy some "executors" that do
    # stuff with the Solver data
    # final_time = 4 * dt
    while t <= final_time:
        if animate and (len(data) - 1 < t // time_interval or t == 0 or allFrames):
            print("save frame")
            cells = solver.mesh.cells
            data.append(np.array(cells.values[..., 0, Q.fields.pbar]))

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

        rhoU = cells.values[..., 0, Q.fields.rhoU]
        im.set_data(rhoU.transpose())

        plt.tight_layout()
        plt.show()
        plt.close()

    if animate:
        min_var = min(np.min(data[frame]) for frame in np.arange(0, len(data)))
        max_var = max(np.max(data[frame]) for frame in np.arange(0, len(data)))
        im.set_clim(min_var, max_var)

        def update(frame):
            im.set_data(data[frame].transpose())
            return ax, im

        _ = FuncAnimation(
            fig,
            update,
            frames=np.arange(0, len(data)),
            init_func=init,
            blit=True,
            interval=200,
        )
        plt.show()
