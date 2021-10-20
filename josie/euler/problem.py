# josiepy
# Copyright © 2019 Ruben Di Battista
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


from josie.dimension import MAX_DIMENSIONALITY
from josie.math import Direction
from josie.mesh.cellset import CellSet
from josie.problem import Problem

from .eos import EOS
from .state import EulerState
from .fields import ConsFields


class EulerProblem(Problem):
    """A class representing an Euler system problem

    Attributes
    ---------
    eos
        An instance of :class:`~.EOS`, an equation of state for the fluid
    """

    def __init__(self, eos: EOS, **kwargs):
        super().__init__(**kwargs)

        self.eos = eos

    def F(self, cells: CellSet) -> np.ndarray:
        r""" This returns the tensor representing the flux for an Euler model

        A general problem can be written in a compact way:

        .. math::

            \pdv{\vb{q}}{t} + \div{\vb{F\qty(\vb{q})}} + \vb{B}\qty(\vb{q})
            \cdot \gradient{\vb{q}} = \vb{s\qty(\vb{q})}

        This function needs to return :math:`\vb{F}\qty(\vb{q})`

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` that contains the cell data

        Returns
        ---------
        F
            An array of dimension :math:`Nx \times Ny \times 4 \times 2`, i.e.
            an array that of each :math:`x` cell and :math:`y` cell stores the
            :math:`4 \times 2` flux tensor

            The flux tensor is:

            .. math::

                \pdeConvective =
                \begin{bmatrix}
                    \rho u & \rho v \\
                    \rho u^2 + p & \rho uv \\
                    \rho vu & \rho v^ 2 + p \\
                    (\rho E + p)U & (\rho E + p)V
                \end{bmatrix}
        """
        values: EulerState = cells.values.view(EulerState)
        fields = values.fields

        num_cells_x, num_cells_y, num_dofs, _ = values.shape

        # Flux tensor
        F = np.empty(
            (
                num_cells_x,
                num_cells_y,
                num_dofs,
                len(ConsFields),
                MAX_DIMENSIONALITY,
            )
        )

        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        rhoE = values[..., fields.rhoE]
        U = values[..., fields.U]
        V = values[..., fields.V]
        p = values[..., fields.p]

        rhoUU = np.multiply(rhoU, U)
        rhoUV = np.multiply(rhoU, V)
        rhoVV = np.multiply(rhoV, V)
        rhoVU = rhoUV  # np.multiply(rhoV, U)

        F[..., fields.rho, Direction.X] = rhoU
        F[..., fields.rho, Direction.Y] = rhoV
        F[..., fields.rhoU, Direction.X] = rhoUU + p
        F[..., fields.rhoU, Direction.Y] = rhoUV
        F[..., fields.rhoV, Direction.X] = rhoVU
        F[..., fields.rhoV, Direction.Y] = rhoVV + p
        F[..., fields.rhoE, Direction.X] = np.multiply(rhoE + p, U)
        F[..., fields.rhoE, Direction.Y] = np.multiply(rhoE + p, V)

        return F
