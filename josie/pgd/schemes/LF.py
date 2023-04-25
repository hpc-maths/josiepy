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
from josie.pgd.state import PGDState

from .scheme import PGDScheme

if TYPE_CHECKING:
    from josie.mesh.cellset import NeighboursCellSet, MeshCellSet


class LF(PGDScheme):
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

    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet):

        Q_L: PGDState = cells.values.view(PGDState)
        Q_R: PGDState = neighs.values.view(PGDState)

        fields = PGDState.fields

        FS = np.zeros_like(Q_L).view(PGDState)
        DeltaF = np.zeros_like(self.problem.F(cells)).view(PGDState)

        u_L = Q_L[..., fields.U]
        u_R = Q_R[..., fields.U]
        v_L = Q_L[..., fields.V]
        v_R = Q_R[..., fields.V]

        sigma = max(
            np.amax(np.abs(u_L)),
            np.amax(np.abs(u_R)),
            np.amax(np.abs(v_R)),
            np.amax(np.abs(v_L)),
        )

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Q_L_cons = Q_L.get_conservative()
        Q_R_cons = Q_R.get_conservative()
        DeltaQ = np.zeros_like(Q_L_cons).view(PGDState)

        if neighs.direction == 0:
            DeltaF[..., 0:2, :, :] = 0.5 * (
                self.problem.F(cells)[..., 0:2, :, :]
                + self.problem.F(neighs)[..., 2:4, :, :]
            )
            DeltaQ[..., 0:2, :] = (
                0.5 * sigma * (Q_R_cons[..., 2:4, :] - Q_L_cons[..., 0:2, :])
            )
        if neighs.direction == 1:
            DeltaF[..., 0, :, :] = 0.5 * (
                self.problem.F(cells)[..., 0, :, :]
                + self.problem.F(neighs)[..., 1, :, :]
            )
            DeltaQ[..., 0, :] = (
                0.5 * sigma * (Q_R_cons[..., 1, :] - Q_L_cons[..., 0, :])
            )
            DeltaF[..., 2, :, :] = 0.5 * (
                self.problem.F(cells)[..., 2, :, :]
                + self.problem.F(neighs)[..., 3, :, :]
            )
            DeltaQ[..., 2, :] = (
                0.5 * sigma * (Q_R_cons[..., 3, :] - Q_L_cons[..., 2, :])
            )
        if neighs.direction == 2:
            DeltaF[..., 2:4, :, :] = 0.5 * (
                self.problem.F(cells)[..., 2:4, :, :]
                + self.problem.F(neighs)[..., 0:2, :, :]
            )

            DeltaQ[..., 2:4, :] = (
                0.5 * sigma * (Q_R_cons[..., 0:2, :] - Q_L_cons[..., 2:4, :])
            )
        if neighs.direction == 3:
            DeltaF[..., 1, :, :] = 0.5 * (
                self.problem.F(cells)[..., 1, :, :]
                + self.problem.F(neighs)[..., 0, :, :]
            )
            DeltaQ[..., 1, :] = (
                0.5 * sigma * (Q_R_cons[..., 0, :] - Q_L_cons[..., 1, :])
            )
            DeltaF[..., 3, :, :] = 0.5 * (
                self.problem.F(cells)[..., 3, :, :]
                + self.problem.F(neighs)[..., 2, :, :]
            )
            DeltaQ[..., 3, :] = (
                0.5 * sigma * (Q_R_cons[..., 2, :] - Q_L_cons[..., 3, :])
            )

        Delta = np.einsum("...mkl,...l->...mk", DeltaF, neighs.normals)
        FS.set_conservative(Delta - DeltaQ)
        return FS
