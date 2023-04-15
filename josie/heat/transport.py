# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import numpy as np

from typing import TYPE_CHECKING, Union

from josie.transport import Transport

if TYPE_CHECKING:
    from josie.mesh.cellset import CellSet, MeshCellSet


class HeatTransport(Transport):
    """A class providing the thermal diffusivity for the temperature"""

    @abc.abstractmethod
    def thermal_diffusivity(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        r"""Thermal diffusivity :math:`\thermalDiffusivity`.
        Units: :math:`\qty[\si{\meter \per \square \second}]`

        .. math::

            \alpha =
            \frac{\thermalConductivity}{\density \specificHeat_\pressure}
        """
        raise NotImplementedError


class ConstantHeatTransport(HeatTransport):
    r"""A :class:`HeatTransport` providing constant
    :math:`\thermalDiffusivity`

    Parameters
    ----------
    thermal_diffusivity
        The constant value of the thermal diffusivity
    """

    def __init__(self, thermal_diffusivity: float):
        self._thermal_diffusivity = thermal_diffusivity

    def thermal_diffusivity(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        nx, ny, num_dofs, _ = cells.values.shape

        return np.ones((nx, ny, num_dofs)) * self._thermal_diffusivity
