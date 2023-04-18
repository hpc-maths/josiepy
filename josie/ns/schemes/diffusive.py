# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

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

    # problem: NSProblem

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
