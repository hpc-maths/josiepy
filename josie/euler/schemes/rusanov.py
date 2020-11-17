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
from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from josie.euler.state import Q

from .scheme import EulerScheme

if TYPE_CHECKING:
    from josie.mesh.cellset import CellSet, MeshCellSet


class Rusanov(EulerScheme):
    r"""This class implements the Rusanov scheme. See
    :cite:`toro_riemann_2009` for a detailed view on compressible schemes.
    The Rusanov scheme is discretized by:

    .. math::

        \numConvective  =
            \frac{1}{2} \qty[%
            \qty|\pdeConvective|_{i+1} + \qty|\pdeConvective|_{i}
            - \sigma \qty(\pdeState_{i+1} - \pdeState_{i})
            ] S_f
    """

    def F(self, cells: MeshCellSet, neighs: CellSet):

        Q_L: Q = cells.values
        Q_R: Q = neighs.values

        fields = Q.fields

        FS = np.zeros_like(Q_L).view(Q)

        # Get normal velocities
        U_L = self.compute_U_norm(Q_L, neighs.normals)
        U_R = self.compute_U_norm(Q_R, neighs.normals)

        # Speed of sound
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]

        sigma = self.compute_sigma(U_L, U_R, c_L, c_R)

        DeltaF = 0.5 * (self.problem.F(cells) + self.problem.F(neighs))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...kl,...l->...k", DeltaF, neighs.normals)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Q_L_cons = Q_L.get_conservative()
        Q_R_cons = Q_R.get_conservative()

        DeltaQ = 0.5 * sigma * (Q_R_cons - Q_L_cons)

        FS.set_conservative(
            neighs.surfaces[..., np.newaxis] * (DeltaF - DeltaQ)
        )

        return FS
