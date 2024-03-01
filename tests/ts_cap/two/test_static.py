# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import logging
from datetime import datetime
from josie.io.write.writer import XDMFWriter
from josie.io.write.strategy import TimeStrategy, IterationStrategy


from josie.boundary import Line
from josie.bc import make_periodic, Direction
from josie.mesh import Mesh
from josie.mesh.cell import MUSCLCell
from josie.mesh.cellset import MeshCellSet
from josie.ts_cap.state import Q
from josie.bn.eos import TwoPhaseEOS
from josie.FourEq.eos import LinearizedGas

from josie.euler.eos import PerfectGas, StiffenedGas
from josie.twofluid.fields import Phases

from tests.ts_cap.two.conftest import circle, square


def test_static(plot, write, request, init_schemes, init_solver, nSmoothPass):
    L = 0.75
    left = Line([0, 0], [0, L])
    bottom = Line([0, 0], [L, 0])
    right = Line([L, 0], [L, L])
    top = Line([0, L], [L, L])

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    N = 80
    mesh.interpolate(N, N)
    mesh.generate()

    nSmoothPass = 10
    sigma = 30
    Hmax = 1e3
    kappa = 1
    dx = mesh.cells._centroids[1, 1, 0, 0] - mesh.cells._centroids[0, 1, 0, 0]
    dy = mesh.cells._centroids[1, 1, 0, 1] - mesh.cells._centroids[1, 0, 0, 1]
    norm_grada_min = 0.01 * 1 / dx
    norm_grada_min = 0

    eos_ref = TwoPhaseEOS(
        phase1=StiffenedGas(gamma=2.1, p0=1e6),
        phase2=PerfectGas(gamma=1.4),
    )
    p_init = 1e5
    rho_liq = 1e3
    rho_gas = 1e0
    eos = TwoPhaseEOS(
        phase1=LinearizedGas(
            p0=p_init,
            rho0=rho_liq,
            c0=eos_ref[Phases.PHASE1].sound_velocity(rho_liq, p_init),
            # c0=1e1,
        ),
        phase2=LinearizedGas(
            p0=p_init,
            rho0=rho_gas,
            c0=eos_ref[Phases.PHASE2].sound_velocity(rho_gas, p_init),
            # c0=1e1,
        ),
    )

    schemes = init_schemes(eos, sigma, Hmax, kappa, dx, dy, norm_grada_min, nSmoothPass)

    def init_fun(cells: MeshCellSet):
        # include ghost cells
        x_c = cells._centroids[..., 0]
        y_c = cells._centroids[..., 1]
        x_0 = L / 2
        y_0 = L / 2

        ad = 0
        U_0 = 0
        U_1 = 0
        V = 0
        fields = Q.fields
        R = 0.15

        # Mollifier
        abar = square(R, x_c, y_c, x_0, y_0)

        # No small-scale
        ad = 0
        capSigma = 0

        # Update geometry
        cells._values[..., fields.abar] = abar
        schemes[0].updateGeometry(cells._values)

        # Adjust pressure in the droplet
        H = cells._values[..., fields.H]

        # Enforce mixture smooth transition
        pbar = eos[Phases.PHASE2].p0 + abar * sigma / R
        p1 = np.full_like(abar, np.nan)
        p1 = np.where(
            abar == 1,
            eos[Phases.PHASE2].p0 + sigma / R,
            np.where((abar < 1) & (abar > 0), pbar + (1 - abar) * sigma * H, np.nan),
        )
        p1 = p_init
        rho1 = eos[Phases.PHASE1].rho(p1)
        p2 = np.full_like(abar, np.nan)
        p2 = np.where(
            abar == 0,
            eos[Phases.PHASE2].p0,
            np.where((abar < 1) & (abar > 0), pbar - abar * sigma * H, np.nan),
        )
        p2 = p_init
        rho2 = eos[Phases.PHASE2].rho(p2)

        # Compute conservative variables
        arho1 = np.zeros_like(abar)
        arho1 = np.where((abar > 0) & ((~np.isnan(rho1))), rho1 * abar * (1 - ad), 0)
        arho2 = np.where(
            (abar < 1) & ((~np.isnan(rho2))), rho2 * (1 - abar) * (1 - ad), 0
        )
        arho1d = eos[Phases.PHASE1].rho0 * ad
        rho = arho1 + arho2 + arho1d
        U = abar * U_1 + (1 - abar) * U_0
        rhoU = rho * U
        rhoV = rho * V

        cells._values[..., fields.abarrho] = abar * rho
        cells._values[..., fields.ad] = ad
        cells._values[..., fields.capSigma] = capSigma
        cells._values[..., fields.arho1] = arho1
        cells._values[..., fields.arho1d] = arho1d
        cells._values[..., fields.arho2] = arho2
        cells._values[..., fields.rhoU] = rhoU
        cells._values[..., fields.rhoV] = rhoV
        schemes[0].auxilliaryVariableUpdate(cells._values)

    solver = init_solver(mesh, schemes, init_fun)

    final_time = 1
    final_time_test = 1e-4
    CFL = 0.4
    if write:
        now = datetime.now().strftime("%Y%m%d%H%M%S")

        logger = logging.getLogger("josie")
        logger.setLevel(logging.DEBUG)

        test_name = request.node.name.replace("[", "-").replace("]", "-")
        fh = logging.FileHandler(test_name + f"{now}.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        # Write strategy
        strategy = TimeStrategy(dt_save=final_time / 200, animate=False)
        writer = XDMFWriter(
            test_name + f"{now}.xdmf", strategy, solver, final_time=final_time, CFL=CFL
        )

        writer.solve()

    else:
        final_time = final_time_test
        t = 0.0
        while t <= final_time:
            dt = solver.CFL(CFL)

            assert ~np.isnan(dt)
            solver.step(dt)

            t += dt
            print(f"Time: {t}, dt: {dt}")

    # Assert symmetry
    data = solver.mesh.cells.values[..., 0, Q.fields.abarrho]
    xSymNorm = np.linalg.norm(data - data[::-1, ...]) / np.linalg.norm(data)
    ySymNorm = np.linalg.norm(data - data[:, ::-1, ...]) / np.linalg.norm(data)
    xySymNorm = np.linalg.norm(data - np.transpose(data, (0, 1))) / np.linalg.norm(data)
    print("X symmetry : " + str(xSymNorm))
    print("Y symmetry : " + str(ySymNorm))
    print("XY symmetry : " + str(xySymNorm))
    assert xSymNorm < 1e-6
    assert ySymNorm < 1e-6
    assert xySymNorm < 1e-6
