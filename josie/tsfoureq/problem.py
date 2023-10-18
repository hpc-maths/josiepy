# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


from ..dimension import MAX_DIMENSIONALITY
from ..problem import ConvectiveProblem
from ..math import Direction

from .eos import TwoPhaseEOS
from .state import Q, TSFourEqConsFields


class TSFourEqProblem(ConvectiveProblem):
    """A class representing a two-phase system problem governed by the
    four equations model (barotropic EOS with velocity equilibrium)"""

    def __init__(self, eos: TwoPhaseEOS):
        self.eos = eos

    def F(self, values: Q) -> np.ndarray:
        r""" """
        num_cells_x, num_cells_y, num_dofs, _ = values.shape

        # Flux tensor
        F = np.zeros(
            (
                num_cells_x,
                num_cells_y,
                num_dofs,
                len(TSFourEqConsFields),
                MAX_DIMENSIONALITY,
            )
        )
        fields = Q.fields

        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho = values[..., fields.arho]
        rho = arho1 + arho2
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        U = rhoU / rho
        V = rhoV / rho
        p = values[..., fields.P]

        F[..., TSFourEqConsFields.arho, Direction.X] = arho * U
        F[..., TSFourEqConsFields.arho, Direction.Y] = arho * V

        F[..., TSFourEqConsFields.arho1, Direction.X] = arho1 * U
        F[..., TSFourEqConsFields.arho1, Direction.Y] = arho1 * V

        F[..., TSFourEqConsFields.arho2, Direction.X] = arho2 * U
        F[..., TSFourEqConsFields.arho2, Direction.Y] = arho2 * V

        F[..., TSFourEqConsFields.rhoU, Direction.X] = rhoU * U + p
        F[..., TSFourEqConsFields.rhoU, Direction.Y] = rhoU * V
        F[..., TSFourEqConsFields.rhoV, Direction.X] = rhoV * U
        F[..., TSFourEqConsFields.rhoV, Direction.Y] = rhoV * V + p

        return F
