# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


from ..dimension import MAX_DIMENSIONALITY
from ..problem import ConvectiveProblem
from ..math import Direction

from .eos import TwoPhaseEOS
from .state import Q, TsCapConsFields


class TsCapProblem(ConvectiveProblem):
    """A class representing a two-phase system problem governed by a
    four equation like model with capillarity"""

    def __init__(
        self,
        eos: TwoPhaseEOS,
        sigma: float,
        Hmax: float,
        norm_grada_min: float,
    ):
        self.eos = eos
        self.sigma = sigma
        self.Hmax = Hmax
        self.norm_grada_min = norm_grada_min

    def F(self, values: Q) -> np.ndarray:
        r""" """
        num_cells_x, num_cells_y, num_dofs, _ = values.shape

        # Flux tensor
        F = np.zeros(
            (
                num_cells_x,
                num_cells_y,
                num_dofs,
                len(TsCapConsFields),
                MAX_DIMENSIONALITY,
            )
        )
        fields = Q.fields

        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho1d = values[..., fields.arho1d]
        abarrho = values[..., fields.abarrho]
        rho = arho1 + arho2 + arho1d
        ad = values[..., fields.ad]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        U = rhoU / rho
        V = rhoV / rho
        pbar = values[..., fields.pbar]
        capSigma = values[..., fields.capSigma]
        norm_grada = values[..., fields.norm_grada]
        n_x = values[..., fields.n_x]
        n_y = values[..., fields.n_y]

        # Four-eq like model
        F[..., TsCapConsFields.abarrho, Direction.X] = abarrho * U
        F[..., TsCapConsFields.abarrho, Direction.Y] = abarrho * V

        F[..., TsCapConsFields.arho1, Direction.X] = arho1 * U
        F[..., TsCapConsFields.arho1, Direction.Y] = arho1 * V

        F[..., TsCapConsFields.arho2, Direction.X] = arho2 * U
        F[..., TsCapConsFields.arho2, Direction.Y] = arho2 * V

        F[..., TsCapConsFields.rhoU, Direction.X] = rhoU * U + pbar
        F[..., TsCapConsFields.rhoU, Direction.Y] = rhoU * V
        F[..., TsCapConsFields.rhoV, Direction.X] = rhoV * U
        F[..., TsCapConsFields.rhoV, Direction.Y] = rhoV * V + pbar

        # Large-scale capillarity
        F[..., TsCapConsFields.rhoU, Direction.X] += np.where(
            norm_grada > self.norm_grada_min,
            self.sigma * norm_grada * (n_x**2 - 1),
            0,
        )
        F[..., TsCapConsFields.rhoU, Direction.Y] += np.where(
            norm_grada > self.norm_grada_min,
            self.sigma * norm_grada * n_x * n_y,
            0,
        )
        F[..., TsCapConsFields.rhoV, Direction.X] += np.where(
            norm_grada > self.norm_grada_min,
            self.sigma * norm_grada * n_y * n_x,
            0,
        )
        F[..., TsCapConsFields.rhoV, Direction.Y] += np.where(
            norm_grada > self.norm_grada_min,
            self.sigma * norm_grada * (n_y**2 - 1),
            0,
        )

        # Small-scale variables
        F[..., TsCapConsFields.arho1d, Direction.X] = arho1d * U
        F[..., TsCapConsFields.arho1d, Direction.Y] = arho1d * V

        F[..., TsCapConsFields.ad, Direction.X] = ad * U
        F[..., TsCapConsFields.ad, Direction.Y] = ad * V

        F[..., TsCapConsFields.capSigma, Direction.X] = capSigma * U
        F[..., TsCapConsFields.capSigma, Direction.Y] = capSigma * V
        return F
