# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" :class:`BoundaryCondition` classes that are specific two-phase systeme 
with capillarity"""
from __future__ import annotations

from typing import TYPE_CHECKING
from josie.bc import BoundaryCondition, Dirichlet, Neumann
from josie.bn.eos import TwoPhaseEOS
from josie.twofluid.fields import Phases
from josie.ts_cap.state import TsCapFields as fields

import numpy as np


if TYPE_CHECKING:
    from josie.bc import ImposedValue
    from josie.mesh.cellset import MeshCellSet
    from josie.boundary import Boundary


class Inlet(BoundaryCondition):
    """Imposes an inlet boundary condition i.e. velocity a
    zero gradient pressure condition is set.

    Parameters
    ----------
    U
        x-component of the flow velocity to impose, as a :class:`Callable`
        taking  :class:`CellSet` instance as parameter referring to the cells
        of the related boundary
    V
        y-component of the flow velocity to impose, as a :class:`Callable`
        taking  :class:`CellSet` as parameter referring to the cells of the
        related boundary

    eos
        The same :class:`TwoPhaseEOS` used for the rest of the problem

    constant
        Set this flag to ``True`` to explicitly force the creation of a
        constant boundary condition. A constant BC is optimized to reduce the
        number of calls.

    """

    # TODO: Add 3D
    def __init__(
        self,
        abar: ImposedValue,
        ad: ImposedValue,
        rho1d: ImposedValue,
        capSigma: ImposedValue,
        U: ImposedValue,
        V: ImposedValue,
        eos: TwoPhaseEOS,
        constant=True,
    ):
        # The partial set of BCs to impose
        self.U_bc = Dirichlet(U, constant)
        self.V_bc = Dirichlet(V, constant)
        self.abar_bc = Dirichlet(abar, constant)
        self.ad_bc = Dirichlet(ad, constant)
        self.rho1d_bc = Dirichlet(rho1d, constant)
        self.capSigma_bc = Dirichlet(capSigma, constant)

        self.zero_gradient = Neumann(0, constant)

        self.eos = eos

    def init(self, cells: MeshCellSet, boundary: Boundary):
        boundary_idx = boundary.cells_idx
        boundary_cells = cells[boundary_idx]

        self.U_bc.init(boundary_cells)
        self.V_bc.init(boundary_cells)
        self.abar_bc.init(boundary_cells)
        self.ad_bc.init(boundary_cells)
        self.rho1d_bc.init(boundary_cells)
        self.capSigma_bc.init(boundary_cells)

        self.zero_gradient.init(boundary_cells)

    def __call__(self, cells: MeshCellSet, boundary: Boundary, t: float):
        ghost_idx = boundary.ghost_cells_idx
        boundary_idx = boundary.cells_idx

        boundary_cells = cells[boundary_idx]
        ghost_cells = cells[ghost_idx]

        # Let's compute the ghost value of pressure that imposes the zero
        # gradient condition
        p1_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.p1, t)
        p2_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.p2, t)

        # dirichlet conditions
        abar_ghost = self.abar_bc(boundary_cells, ghost_cells, fields.abar, t)
        ad_ghost = self.ad_bc(boundary_cells, ghost_cells, fields.ad, t)
        rho1d_ghost = self.ad_bc(boundary_cells, ghost_cells, fields.ad, t)
        capSigma_ghost = self.capSigma_bc(
            boundary_cells, ghost_cells, fields.capSigma, t
        )
        U_ghost = self.U_bc(boundary_cells, ghost_cells, fields.U, t)
        V_ghost = self.V_bc(boundary_cells, ghost_cells, fields.V, t)

        # Compute densities
        rho1_ghost = self.eos[Phases.PHASE1].rho(p1_ghost)
        rho2_ghost = self.eos[Phases.PHASE2].rho(p2_ghost)
        c1_ghost = self.eos[Phases.PHASE1].sound_velocity(rho1_ghost)
        c2_ghost = self.eos[Phases.PHASE2].sound_velocity(rho2_ghost)
        arho1_ghost = np.where(
            ~np.isnan(rho1_ghost), abar_ghost * (1 - ad_ghost) * rho1_ghost, 0
        )
        arho2_ghost = np.where(
            ~np.isnan(rho2_ghost), (1 - abar_ghost) * (1 - ad_ghost) * rho2_ghost, 0
        )
        arho1d_ghost = ad_ghost * rho1d_ghost
        rho_ghost = arho1_ghost + arho2_ghost + arho1d_ghost

        # Compute conserved variables
        abarrho_ghost = abar_ghost * rho_ghost
        rhoU_ghost = rho_ghost * U_ghost
        rhoV_ghost = rho_ghost * V_ghost

        # Compute auxiliary variables
        pbar_ghost = np.where(
            (~np.isnan(p1_ghost)) & (~np.isnan(p2_ghost)),
            abar_ghost * p1_ghost + (1 - abar_ghost) * p2_ghost,
            np.where(
                ~np.isnan(p1_ghost), abar_ghost * p1_ghost, (1 - abar_ghost) * p2_ghost
            ),
        )
        cFd_ghost = np.where(
            (~np.isnan(p1_ghost)) & (~np.isnan(p2_ghost)),
            arho1_ghost * c1_ghost**2 + arho2_ghost * c2_ghost**2,
            np.where(
                ~np.isnan(p1_ghost),
                arho1_ghost * c1_ghost**2,
                arho2_ghost * c2_ghost**2,
            ),
        )
        cFd_ghost = np.sqrt(cFd_ghost / rho_ghost) / (1 - ad_ghost)

        # geometry variables are set to NaN

        # Impose the ghost values
        for field, ghost_value in (
            (fields.abarrho, abarrho_ghost),
            (fields.rhoU, rhoU_ghost),
            (fields.rhoV, rhoV_ghost),
            (fields.rho, rho_ghost),
            (fields.U, U_ghost),
            (fields.V, V_ghost),
            (fields.pbar, pbar_ghost),
            (fields.cFd, cFd_ghost),
            (fields.abar, abar_ghost),
            (fields.arho1, arho1_ghost),
            (fields.p1, p1_ghost),
            (fields.c1, c1_ghost),
            (fields.arho2, arho2_ghost),
            (fields.p2, p2_ghost),
            (fields.c2, c2_ghost),
            (fields.arho1d, arho1d_ghost),
            (fields.ad, ad_ghost),
            (fields.capSigma, capSigma_ghost),
            (fields.grada_x, np.nan),
            (fields.grada_y, np.nan),
            (fields.n_x, np.nan),
            (fields.n_y, np.nan),
            (fields.norm_grada, np.nan),
            (fields.H, np.nan),
            (fields.MaX, np.nan),
            (fields.MaY, np.nan),
            (fields.WeX, np.nan),
            (fields.WeY, np.nan),
            (fields.c_cap1X, np.nan),
            (fields.c_cap1Y, np.nan),
            (fields.c_cap2X, np.nan),
            (fields.c_cap2Y, np.nan),
        ):
            cells._values[ghost_idx[0], ghost_idx[1], :, field] = ghost_value
