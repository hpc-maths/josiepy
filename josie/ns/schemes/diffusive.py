# josiepy
# Copyright © 2021 Ruben Di Battista
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
from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from josie.general.schemes.diffusive import CentralDifferenceGradient as _CDG
from josie.ns.state import NSGradientState, NSState

if TYPE_CHECKING:
    from josie.mesh.cellset import MeshCellSet, NeighboursCellSet
    from josie.ns.problem import NSProblem


class CentralDifferenceGradient(_CDG):
    """Optimizing the implementation of the
    :class:`~general.schemes.diffusive.CentralDifferenceGradient` using a
    viscosity tensor that is smaller in size noting that it only operates on
    the fields :math:`u, v, e`"""

    problem: NSProblem

    def D(self, cells: MeshCellSet, neighs: NeighboursCellSet):
        values: NSState = cells.values.view(NSState)

        fields = values.fields

        # Retrieve neighbour index
        idx = self._directions[neighs.direction]

        # Retrieve length of the relative vector between cell and neighbour
        r = self._r[..., idx, np.newaxis, np.newaxis]

        # Estimate the gradient in normal direction acting only on the gradient
        # variables
        Q_L = cells.values.view(NSState).get_diffusive().view(NSGradientState)
        Q_R = neighs.values.view(NSState).get_diffusive().view(NSGradientState)

        dQ = (Q_R - Q_L) / r

        KdQ = np.einsum("...ijkl,...j->...i", self.problem.K(cells), dQ)

        # Multiply by the surface
        KdQS = KdQ * neighs.surfaces[..., np.newaxis, np.newaxis]

        # The diffusive contribution to the flux concerns only the equations on
        # rhoU and rhoV
        D = np.zeros_like(self._fluxes)
        D[..., fields.rhoU : fields.rhoE + 1] = KdQS

        return D
