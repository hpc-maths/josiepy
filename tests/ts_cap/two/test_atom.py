# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import logging
from datetime import datetime
from josie.io.write.writer import XDMFWriter
from josie.io.write.strategy import TimeStrategy


from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import MUSCLCell
from josie.mesh.cellset import MeshCellSet
from josie.ts_cap.state import Q
from josie.bn.eos import TwoPhaseEOS
from josie.FourEq.eos import LinearizedGas

from josie.ts_cap.bc import Inlet
from josie.bc import Neumann

from dataclasses import dataclass
from .conftest import circle


from josie.twofluid.fields import Phases


@dataclass
class AtomParam:
    name: str
    We: float
    sigma: float
    rho0: float
    final_time: float
    final_time_test: float


atom_params = [
    # AtomParam(
    #     name="No cap",
    #     We=100,
    #     sigma=1e-2,
    #     rho0=1e1,
    #     final_time=3,
    #     final_time_test=1e-2,
    # ),
    AtomParam(
        name="Sheet_stripping",
        We=100,
        sigma=1e-2,
        rho0=1e3,
        final_time=3,
        final_time_test=1e-2,
    ),
    # AtomParam(
    #     name="We 5",
    #     We=5,
    #     sigma=1e-2,
    #     rho0=1e1,
    #     final_time=3,
    #     final_time_test=3e-2,
    # ),
    # AtomParam(
    #     name="We 10",
    #     We=10,
    #     sigma=1e-2,
    #     rho0=1e1,
    #     final_time=3,
    #     final_time_test=3e-2,
    # ),
    # AtomParam(
    #     name="We 50",
    #     We=50,
    #     sigma=1e-2,
    #     rho0=1e1,
    #     final_time=3,
    #     final_time_test=3e-2,
    # ),
]


@pytest.mark.parametrize(
    "atom_param", atom_params, ids=[atom_param.name for atom_param in atom_params]
)
def test_atom(write, init_schemes, init_solver, atom_param):
    box_ratio = 2
    height = 2
    width = box_ratio * height

    left = Line([0, 0], [0, height])
    bottom = Line([0, 0], [width, 0])
    right = Line([width, 0], [width, height])
    top = Line([0, height], [width, height])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e5, rho0=atom_param.rho0, c0=1e1),
        phase2=LinearizedGas(p0=1e5, rho0=1e0, c0=1e1),
    )

    # Initial conditions
    We = atom_param.We  # We = rho * (U_l-U_g) * L / sigma
    sigma = atom_param.sigma
    R = 0.2
    nSmoothPass = 10

    U_inlet = We / eos[Phases.PHASE2].rho0 / R * sigma

    # Inlet BC
    abar_in = 0
    ad_in = 0
    rho1d_in = eos[Phases.PHASE1].rho0
    capSigma_in = 0
    U_in = U_inlet
    V_in = 0

    # left, right = make_periodic(left, right, Direction.X)
    # bottom, top = make_periodic(bottom, top, Direction.Y)
    left.bc = Inlet(abar_in, ad_in, rho1d_in, capSigma_in, U_in, V_in, eos)
    right.bc = Neumann(np.zeros(len(Q.fields)).view(Q))
    bottom.bc = Neumann(np.zeros(len(Q.fields)).view(Q))
    top.bc = Neumann(np.zeros(len(Q.fields)).view(Q))

    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    N = 80
    mesh.interpolate(int(box_ratio * N), N)
    mesh.generate()

    Hmax = 1e3
    dx = mesh.cells._centroids[1, 1, 0, 0] - mesh.cells._centroids[0, 1, 0, 0]
    dy = mesh.cells._centroids[1, 1, 0, 1] - mesh.cells._centroids[1, 0, 0, 1]
    norm_grada_min = 0.05 * 1 / dx

    schemes = init_schemes(eos, sigma, Hmax, dx, dy, norm_grada_min, nSmoothPass)

    def mollify_state(cells, r, ad, U_0, U_1, V, x_c, y_c, x_0, y_0):
        fields = Q.fields

        # Mollifier
        w = circle(R, x_c, y_c, x_0, y_0, False)

        # No small-scale
        ad = 0
        capSigma = 0

        # Update geometry
        abar = w
        cells._values[..., fields.abar] = abar
        schemes[0].updateGeometry(cells._values)

        # Adjust pressure in the droplet
        H = cells._values[..., fields.H]
        p1 = np.full_like(abar, np.nan)
        p1 = np.where(
            abar == 1,
            eos[Phases.PHASE2].p0 + sigma / R,
            np.where(
                (abar < 1) & (abar > 0), eos[Phases.PHASE2].p0 + sigma * H, np.nan
            ),
        )
        rho1 = eos[Phases.PHASE1].rho(p1)

        # Compute conservative variables
        arho1 = np.zeros_like(abar)
        arho1 = np.where((abar > 0) & ((~np.isnan(rho1))), rho1 * abar * (1 - ad), 0)
        arho2 = eos[Phases.PHASE2].rho0 * (1 - abar) * (1 - ad)
        arho1d = eos[Phases.PHASE1].rho0 * ad
        rho = arho1 + arho2 + arho1d
        rhoU = arho1 * U_1 + arho2 * U_0
        U = rhoU / rho
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

    def init_fun(cells: MeshCellSet):
        # include ghost cells
        x_c = cells._centroids[..., 0]
        y_c = cells._centroids[..., 1]
        x_0 = width / 4
        y_0 = height / 2

        ad = 0
        U_0 = U_inlet
        U_1 = 0
        V = 0

        r = np.sqrt((x_c - x_0) ** 2 + (y_c - y_0) ** 2)
        mollify_state(cells, r, ad, U_0, U_1, V, x_c, y_c, x_0, y_0)
        schemes[0].auxilliaryVariableUpdate(cells._values)

    solver = init_solver(mesh, schemes, init_fun)

    CFL = 0.4
    if write:
        final_time = atom_param.final_time
        now = datetime.now().strftime("%Y%m%d%H%M%S")

        logger = logging.getLogger("josie")
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(
            atom_param.name + "-" + str(atom_param.We) + f"-{now}.log"
        )
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        # Write strategy
        dt_save = final_time / 100
        strategy = TimeStrategy(dt_save=dt_save, animate=False)
        writer = XDMFWriter(
            atom_param.name + "-" + str(atom_param.We) + f"-{now}.xdmf",
            strategy,
            solver,
            final_time=final_time,
            CFL=CFL,
        )

        writer.solve()
    else:
        final_time = atom_param.final_time_test
        t = 0.0
        while t <= final_time:
            dt = solver.CFL(CFL)

            assert ~np.isnan(dt)
            solver.step(dt)

            t += dt
            print(f"Time: {t}, dt: {dt}")
