# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.scheme.convective import ConvectiveScheme
from josie.dimension import MAX_DIMENSIONALITY
from josie.mesh.cellset import MeshCellSet

from josie.advection.state import Q
from josie.advection.problem import AdvectionProblem


class AdvectionScheme(ConvectiveScheme):
    problem: AdvectionProblem

    def CFL(
        self,
        cells: MeshCellSet,
        CFL_value: float,
    ) -> float:
        U_abs = np.linalg.norm(self.problem.V)
        dx = np.min(cells.surfaces)

        return CFL_value * dx / U_abs


class Upwind(AdvectionScheme):
    def intercellFlux(self, Q_L: Q, Q_R: Q, normals: np.ndarray, surfaces: np.ndarray):
        nx, ny, num_dofs, num_fields = Q_L.shape

        FS = np.zeros_like(Q_L)
        F = np.zeros((nx, ny, num_dofs, num_fields, MAX_DIMENSIONALITY))

        # Dot product of each normal in `norm` by the advection velocity
        # Equivalent to: un = np.sum(Advection.V*(normals), axis=-1)
        Vn = np.einsum("...k,k->...", normals, self.problem.V)

        # Check where un > 0
        idx = np.where(Vn > 0)

        if np.any(np.nonzero(idx)):
            F[idx] = self.problem.flux(Q_L)[idx]

        idx = np.where(Vn < 0)
        if np.any(np.nonzero(idx)):
            F[idx] = self.problem.flux(Q_R)[idx]

        FS = (
            np.einsum("...mkl,...l->...mk", F, normals)
            * surfaces[..., np.newaxis, np.newaxis]
        )

        return FS
