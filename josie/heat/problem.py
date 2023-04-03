# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, Union

from josie.fluid.problem import DiffusiveProblem

if TYPE_CHECKING:
    from josie.mesh.cellset import MeshCellSet, CellSet
    from .transport import HeatTransport


class HeatProblem(DiffusiveProblem):
    """A class representing a system governed by the heat equation

    Attributes
    ----------
    transport
        A instance of :class:`HeatTransport` providing transport coefficients

    """

    transport: HeatTransport

    def __init__(self, transport: HeatTransport):
        super().__init__(transport=transport)

    def K(self, cells: Union[CellSet, MeshCellSet]) -> np.ndarray:
        """This is a scalar value, but :class:`DiffusiveProblem` expects a 4th
        rank tensor"""
        nx, ny, num_dofs, num_fields = cells.values.shape
        dimensionality = cells.dimensionality

        return self.transport.thermal_diffusivity(cells).reshape(
            (
                nx,
                ny,
                num_dofs,
                num_fields,
                num_fields,
                dimensionality,
                dimensionality,
            )
        )
