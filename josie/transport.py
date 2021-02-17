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

from typing import TYPE_CHECKING, Union

import abc
import numpy as np

if TYPE_CHECKING:
    from josie.mesh.cellset import CellSet, MeshCellSet


class Transport(abc.ABC):
    """A class providing the transport coefficients

    Parameters
    ----------
    cells
        A reference to :class:`MeshCellSet`
    """

    @abc.abstractmethod
    def momentum_diffusivity(self, cells: Union[MeshCellSet, CellSet]):
        r"""Momentum diffusivity also called kinematic viscosity
        :math:`\kinematicViscosity`.  Units: :math:`\qty[\si{\meter \per
        \square \second}]`

        .. math::

            \kinematicViscosity = \frac{\viscosity}{\density}

        being :math:`\viscosity} the dynamic viscosity

        """
        raise NotImplementedError

    @abc.abstractmethod
    def thermal_diffusivity(self, cells: Union[MeshCellSet, CellSet]):
        r"""Thermal diffusivity :math:`\thermalDiffusivity`.
        Units: :math:`\qty[\si{\meter \per \square \second}]`

        .. math::

            \alpha =
            \frac{\thermalConductivity}{\density \specificHeat_\pressure}
        """
        raise NotImplementedError


class ConstantTransport(Transport):
    """A :class:`Transport` providing constant coefficients

    Parameters
    ----------
    momentum_diffusivity
        The constant value of the momentum diffusivity

    thermal_diffusivity
        The constant value of the thermal diffusivity
    """

    def __init__(
        self,
        momentum_diffusivity: float,
        thermal_diffusivity: float,
    ):
        self._momentum_diffusivity = momentum_diffusivity
        self._thermal_diffusivity = thermal_diffusivity

    def momentum_diffusivity(self, cells: Union[MeshCellSet, CellSet]):
        nx, ny, num_fields = cells.values.shape
        dimensionality = cells.dimensionality

        # FIXME: num_fields can be different from the number of fields in the
        # state (`num_state`)

        return (
            np.ones(
                (
                    nx,
                    ny,
                    num_fields,
                    num_fields,
                    dimensionality,
                    dimensionality,
                )
            )
            * self._momentum_diffusivity
        )

    def thermal_diffusivity(self, cells: Union[MeshCellSet, CellSet]):
        nx, ny, num_fields = cells.values.shape

        nx, ny, num_fields = cells.values.shape
        dimensionality = cells.dimensionality

        # FIXME: num_fields can be different from the number of fields in the
        # state (`num_state`)

        return (
            np.ones(
                (
                    nx,
                    ny,
                    num_fields,
                    num_fields,
                    dimensionality,
                    dimensionality,
                )
            )
            * self._thermal_diffusivity
        )
