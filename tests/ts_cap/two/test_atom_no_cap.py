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
from josie.ts_cap.solver import TsCapSolver
from josie.ts_cap.state import Q
from josie.bn.eos import TwoPhaseEOS
from josie.FourEq.eos import LinearizedGas

from josie.ts_cap.schemes import Rusanov
from josie.general.schemes.space.muscl import MUSCL
from josie.general.schemes.space.limiters import MinMod

from josie.general.schemes.time.rk import RK2_relax
from josie.bc import make_periodic, Direction

from dataclasses import dataclass


from josie.twofluid.fields import Phases


class TsCapScheme(Rusanov, RK2_relax, MUSCL, MinMod):
    pass


@dataclass
class AtomParam:
    name: str
    We: float
    sigma: float
    rho0: float
    final_time: float
    final_time_test: float


atom_params = [
    AtomParam(
        name="Sheet stripping",
        We=200,
        sigma=0,
        rho0=1e1,
        final_time=3,
        final_time_test=3e-3,
    ),
]


@pytest.mark.parametrize(
    "atom_param", atom_params, ids=[atom_param.name for atom_param in atom_params]
)
def test_atom_no_cap(plot, request, atom_param):
    box_ratio = 2
    height = 1
    width = box_ratio * height
    left = Line([0, 0], [0, height])
    bottom = Line([0, 0], [width, 0])
    right = Line([width, 0], [width, height])
    top = Line([0, height], [width, height])

    eos = TwoPhaseEOS(
        phase1=LinearizedGas(p0=1e2, rho0=atom_param.rho0, c0=1e1),
        phase2=LinearizedGas(p0=1e2, rho0=1e0, c0=1e1),
    )

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, MUSCLCell)
    N = 111
    mesh.interpolate(int(box_ratio * N), N)
    mesh.generate()

    # Initial conditions
    We = atom_param.We  # We = rho * (U_l-U_g) * L / sigma
    sigma = atom_param.sigma
    R = 0.2

    U_inlet = We / eos[Phases.PHASE1].rho0 / R * sigma

    # Same conditions but no capillarity
    sigma = 0

    Hmax = 1e3
    dx = mesh.cells._centroids[1, 1, 0, 0] - mesh.cells._centroids[0, 1, 0, 0]
    dy = mesh.cells._centroids[1, 1, 0, 1] - mesh.cells._centroids[1, 0, 0, 1]
    norm_grada_min = 0.01 * 1 / dx
    norm_grada_min = 0

    scheme = TsCapScheme(
        eos,
        sigma,
        Hmax,
        dx,
        dy,
        norm_grada_min,
    )

    scheme.tmp_arr = np.zeros((int(box_ratio * N), N, 5))

    def rbf(R: float, r: np.ndarray):
        eps = R / 2

        def f(x):
            return np.maximum(np.exp(2 * x**2 * (x**2 - 3) / (x**2 - 1) ** 2), 0)

        arr = np.where(
            (r > R) * (r < R + eps),
            f((r - R) / eps),
            np.where(r < R, 1, 0),
        )

        # Enforce symmetry along
        # X-axis
        ny = arr.shape[1]
        # arr = 0.5 * (arr + arr[::-1, :])
        # Y-axis
        arr = 0.5 * (arr + arr[:, ::-1])
        # XY-axis
        arr[:ny, :ny] = 0.5 * (
            arr[:ny, :ny] + np.transpose(arr[:ny, :ny], axes=(1, 0, 2))
        )

        return arr

    def mollify_state(cells, r, ad, U_0, U_1, V):
        fields = Q.fields

        # Mollifier
        w = rbf(R, r)

        # No small-scale
        ad = 0
        capSigma = 0

        # Update geometry
        abar = w
        cells._values[..., fields.abar] = abar
        solver.scheme.updateGeometry(cells._values)

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
        x_0 = 0.5
        y_0 = 0.5

        ad = 0
        U_0 = U_inlet
        U_1 = 0
        V = 0

        r = np.sqrt((x_c - x_0) ** 2 + (y_c - y_0) ** 2)
        mollify_state(cells, r, ad, U_0, U_1, V)

    solver = TsCapSolver(mesh, scheme)
    solver.init(init_fun)

    solver.mesh.update_ghosts(0)
    solver.scheme.auxilliaryVariableUpdate(solver.mesh.cells._values)
    solver.mesh.update_ghosts(0)

    # final_time = atom_params.final_time
    final_time = atom_param.final_time_test
    CFL = 0.4

    now = datetime.now().strftime("%Y%m%d%H%M%S")

    logger = logging.getLogger("josie")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(f"droplet-atom-no-cap-{now}.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    # Write strategy
    if final_time == atom_param.final_time:
        dt_save = final_time / 300
    else:
        dt_save = final_time
    strategy = TimeStrategy(dt_save=dt_save, animate=False)
    writer = XDMFWriter(
        f"droplet-atom-no-cap-{now}.xdmf",
        strategy,
        solver,
        final_time=final_time,
        CFL=CFL,
    )

    writer.solve()
