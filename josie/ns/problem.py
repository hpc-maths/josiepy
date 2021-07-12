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
# official policies, either expressed or implied, of Ruben Di Battista.j
from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Union

from josie.dimension import MAX_DIMENSIONALITY
from josie.fluid.problem import DiffusiveProblem
from josie.euler.problem import EulerProblem
from josie.math import Direction

from .fields import NSGradientFields

if TYPE_CHECKING:
    from josie.euler.eos import EOS
    from josie.mesh.cellset import CellSet, MeshCellSet
    from .transport import NSTransport


class NSProblem(EulerProblem, DiffusiveProblem):

    if TYPE_CHECKING:
        transport: NSTransport

    def __init__(self, eos: EOS, transport: NSTransport):
        super().__init__(eos=eos, transport=transport)

    def K(self, cells: Union[CellSet, MeshCellSet]) -> np.ndarray:
        r"""This method returns the diffusive tensor
        :math:`\pdeDiffusiveMultiplier` for the Navier-Stokes system.

        In 2D it's

        .. math::

            \ipdeDiffusiveMultiplier =
            \nsDiffusiveMultiplierXX (xx)
            \nsDiffusiveMultiplierXY (xy)
            \nsDiffusiveMultiplierYX (yx)
            \nsDiffusiveMultiplierYY (yy)

        """

        nx, ny, _ = cells.values.shape

        fields = NSGradientFields

        num_gradient_fields = len(fields)

        K = np.zeros(
            (
                nx,
                ny,
                num_gradient_fields,
                num_gradient_fields,
                MAX_DIMENSIONALITY,
                MAX_DIMENSIONALITY,
            )
        )

        bulk_viscosity = self.transport.bulk_viscosity(cells)
        mu = self.transport.viscosity(cells)
        alpha = self.transport.thermal_diffusivity(cells)

        Kxx = K[..., Direction.X, Direction.X]
        Kxy = K[..., Direction.X, Direction.Y]
        Kyx = K[..., Direction.Y, Direction.X]
        Kyy = K[..., Direction.Y, Direction.Y]

        # TODO: Add 3D (Kxz, Kyz, Kzx, Kzy, Kzz)
        Kxx[..., fields.U, fields.U] = 2 * mu + bulk_viscosity
        Kxx[..., fields.V, fields.V] = mu
        Kxx[..., fields.rhoe, fields.rhoe] = alpha

        Kxy[..., fields.U, fields.V] = bulk_viscosity
        Kxy[..., fields.V, fields.U] = mu

        # Symmetric
        Kyx[..., fields.V, fields.U] = bulk_viscosity
        Kyx[..., fields.U, fields.V] = mu

        Kyy[..., fields.U, fields.U] = mu
        Kyy[..., fields.V, fields.V] = 2 * mu + bulk_viscosity
        Kyy[..., fields.rhoe, fields.rhoe] = alpha

        return K
