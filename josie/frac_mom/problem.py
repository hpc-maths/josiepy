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
from josie.problem import ConvectiveProblem

from .state import Q
from .fields import ConsFields


class FracMomProblem(ConvectiveProblem):
    """A class representing an PGD system problem

    Attributes
    ---------
    eos
        An instance of :class:`~.EOS`, an equation of state for the fluid
    """

    def F(self, cells: CellSet) -> np.ndarray:
        r""" This returns the tensor representing the flux for an PGD model

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
        values: Q = cells.values.view(Q)
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
        U = values[..., fields.U]
        V = values[..., fields.V]
        m0U = np.multiply(values[..., fields.m0], U)
        m0V = np.multiply(values[..., fields.m0], V)
        m12U = np.multiply(values[..., fields.m12], U)
        m12V = np.multiply(values[..., fields.m12], V)
        m1U = values[..., fields.m1U]
        m1V = values[..., fields.m1V]
        m32U = np.multiply(values[..., fields.m32], U)
        m32V = np.multiply(values[..., fields.m32], V)

        m1UU = np.multiply(m1U, U)
        m1UV = np.multiply(m1U, V)
        m1VV = np.multiply(m1V, V)
        m1VU = m1UV

        F[..., fields.m0, Direction.X] = m0U
        F[..., fields.m0, Direction.Y] = m0V
        F[..., fields.m12, Direction.X] = m12U
        F[..., fields.m12, Direction.Y] = m12V
        F[..., fields.m1, Direction.X] = m1U
        F[..., fields.m1, Direction.Y] = m1V
        F[..., fields.m32, Direction.X] = m32U
        F[..., fields.m32, Direction.Y] = m32V
        F[..., fields.m1U, Direction.X] = m1UU
        F[..., fields.m1U, Direction.Y] = m1UV
        F[..., fields.m1V, Direction.X] = m1VU
        F[..., fields.m1V, Direction.Y] = m1VV

        return F
