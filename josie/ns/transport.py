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
        nx, ny, num_fields = cells.values.shape
        return (
            np.ones(
                (
                    nx,
                    ny,
                )
            )
            * self._viscosity
        )

    def bulk_viscosity(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        nx, ny, num_fields = cells.values.shape
        return (
            np.ones(
                (
                    nx,
                    ny,
                )
            )
            * self._bulk_viscosity
        )

    def thermal_diffusivity(
        self, cells: Union[MeshCellSet, CellSet]
    ) -> np.ndarray:
        nx, ny, num_fields = cells.values.shape

        return (
            np.ones(
                (
                    nx,
                    ny,
                )
            )
            * self._thermal_diffusivity
        )
