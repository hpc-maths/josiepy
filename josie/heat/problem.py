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

    if TYPE_CHECKING:
        transport: HeatTransport

    def __init__(self, transport: HeatTransport):
        super().__init__(transport=transport)

    def K(self, cells: Union[CellSet, MeshCellSet]) -> np.ndarray:
        """This is a scalar value, but :class:`DiffusiveProblem` expects a 4th
        rank tensor"""
        nx, ny, num_fields = cells.values.shape
        dimensionality = cells.dimensionality

        return self.transport.thermal_diffusivity(cells).reshape(
            (nx, ny, num_fields, num_fields, dimensionality, dimensionality)
        )
