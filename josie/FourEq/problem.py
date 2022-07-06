# josiepy
# Copyright © 2020 Ruben Di Battista
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Ruben Di Battista ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Ruben Di Battista BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Ruben Di Battista.
import numpy as np


from josie.dimension import MAX_DIMENSIONALITY
from josie.problem import Problem
from josie.math import Direction

from .eos import TwoPhaseEOS
from .state import Q, FourEqConsFields


class FourEqProblem(Problem):
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
                len(FourEqConsFields),
                MAX_DIMENSIONALITY,
            )
        )
        fields = Q.fields

        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho = values[..., fields.arho]
        rho = arho1 + arho2
        alpha1 = arho / rho
        alpha2 = 1.0 - alpha1
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        U = rhoU / rho
        V = rhoV / rho
        p1 = values[..., fields.p1]
        p2 = values[..., fields.p2]
        p = alpha1 * p1 + alpha2 * p2

        F[..., FourEqConsFields.arho, Direction.X] = arho * U
        F[..., FourEqConsFields.arho, Direction.Y] = arho * V

        F[..., FourEqConsFields.arho1, Direction.X] = arho1 * U
        F[..., FourEqConsFields.arho1, Direction.Y] = arho1 * V

        F[..., FourEqConsFields.arho2, Direction.X] = arho2 * U
        F[..., FourEqConsFields.arho2, Direction.Y] = arho2 * V

        F[..., FourEqConsFields.rhoU, Direction.X] = rhoU * U + p
        F[..., FourEqConsFields.rhoU, Direction.Y] = rhoU * V
        F[..., FourEqConsFields.rhoV, Direction.X] = rhoV * U
        F[..., FourEqConsFields.rhoV, Direction.Y] = rhoV * V + p

        return F
