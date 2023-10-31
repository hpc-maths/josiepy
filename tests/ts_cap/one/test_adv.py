# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

import pytest


from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import MUSCLCell
from josie.mesh.cellset import MeshCellSet
from josie.ts_cap.state import Q
from josie.bn.eos import TwoPhaseEOS
from josie.FourEq.eos import LinearizedGas

from josie.bc import make_periodic, Direction


from josie.twofluid.fields import Phases


@pytest.fixture(params=[Direction.X, Direction.Y])
def dir(request):
    yield request.param


def test_adv(plot, init_schemes, init_solver, dir):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=1e3, c0=1e1),
        phase2=LinearizedGas(p0=1e5, rho0=1e0, c0=1e1),
    )

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    N = 25
    if dir == Direction.X:
        mesh.interpolate(N, 1)
    else:
        mesh.interpolate(1, N)
    mesh.generate()

    Hmax = 1e3
    sigma = 1e-2
    dx = mesh.cells._centroids[1, 1, 0, 0] - mesh.cells._centroids[0, 1, 0, 0]
    dy = mesh.cells._centroids[1, 1, 0, 1] - mesh.cells._centroids[1, 0, 0, 1]
    norm_grada_min = 0
    nSmoothPass = 0

    schemes = init_schemes(
        eos,
        sigma,
        Hmax,
        dx,
        dy,
        norm_grada_min,
        nSmoothPass,
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
        x_c = cells.centroids[..., dir]

        x_left = np.where((x_c < 0.75) * (x_c >= 0.25))
        x_right = np.where(1 - ((x_c < 0.75) * (x_c >= 0.25)))

        abar_L = 1
        abar_R = 0
        ad = 0
        if dir == Direction.X:
            U = 1
            V = 0
        else:
            U = 0
            V = 1

        toState(cells, x_left, abar_L, ad, U, V)
        toState(cells, x_right, abar_R, ad, U, V)

        mesh.update_ghosts(0)

        schemes[0].auxilliaryVariableUpdate(cells._values[..., 0, :])
        mesh.update_ghosts(0)

    solver = init_solver(mesh, schemes, init_fun)

    final_time = 1
    t = 0.0
    CFL = 0.8

    cells = solver.mesh.cells

    if plot:
        # Plot initial solution
        abar_init = cells.values[..., 0, Q.fields.abar].copy().flatten()

    while t <= final_time:
        dt = solver.CFL(CFL)

        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        # Plot final step solution
        x = cells.centroids[..., 0, dir].flatten()
        abar = cells.values[..., 0, Q.fields.abar].flatten()

        plt.plot(x, abar_init)
        plt.plot(x, abar)

        plt.show()
        plt.close()
