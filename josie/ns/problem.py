# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Union

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

        nx, ny, num_dofs, _ = cells.values.shape
        dimensionality = cells.dimensionality

        fields = NSGradientFields

        num_gradient_fields = len(fields)

        K = np.zeros(
            (
                nx,
                ny,
                num_dofs,
                num_gradient_fields,
                num_gradient_fields,
                dimensionality,
                dimensionality,
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
