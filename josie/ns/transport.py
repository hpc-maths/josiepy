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


class NSTransport(Transport):
    @abc.abstractmethod
    def thermal_diffusivity(
        self, cells: Union[MeshCellSet, CellSet]
    ) -> np.ndarray:
        r"""Thermal diffusivity :math:`\thermalDiffusivity`.
        Units: :math:`\qty[\si{\square \meter \per \second}]`

        .. math::

            \thermalDiffusivity = \frac{\thermalConductivity}%
            {\specificHeat_\volume \density}

        It returns a value per each cell centroid
        """
        raise NotImplementedError

    @abc.abstractmethod
    def viscosity(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        r"""Momentum diffusivity also called dynamic viscosity
        :math:`\viscosity`.  Units: :math:`\qty[\si{\pascal \second}]`

        It returns a value per each cell centroid
        """
        raise NotImplementedError

    @abc.abstractmethod
    def bulk_viscosity(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        r"""The second viscosity coefficient also called second viscosity or
        bulk viscosity :math:`\bulkViscosity`.  Units: :math:`\qty[\si{\pascal
        \second}]`


        It returns a value per each cell centroid
        """
        raise NotImplementedError


class NSConstantTransport(NSTransport):
    """A :class:`NSTransport` providing constant coefficients

    Parameters
    ----------
    viscosity
        the constant value of the viscosity

    bulk_viscosity
        the constant value of the bulk viscosity

    thermal_diffusivity
        The constant value of the thermal diffusivity
    """

    def __init__(
        self,
        viscosity: float,
        bulk_viscosity: float,
        thermal_diffusivity: float,
    ):
        self._viscosity = viscosity
        self._bulk_viscosity = bulk_viscosity
        self._thermal_diffusivity = thermal_diffusivity

    def viscosity(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        nx, ny, num_dofs, _ = cells.values.shape
        return np.ones((nx, ny, num_dofs)) * self._viscosity

    def bulk_viscosity(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        nx, ny, num_dofs, _ = cells.values.shape
        return np.ones((nx, ny, num_dofs)) * self._bulk_viscosity

    def thermal_diffusivity(
        self, cells: Union[MeshCellSet, CellSet]
    ) -> np.ndarray:
        nx, ny, num_dofs, _ = cells.values.shape

        return np.ones((nx, ny, num_dofs)) * self._thermal_diffusivity
