# josiepy
# Copyright © 2020 Ruben Di Battista
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
import numpy as np

from typing import Union

from josie._dim import MAX_DIMENSIONALITY
from josie.solver.problem import Problem
from josie.math import Direction
from josie.mesh.cellset import CellSet, MeshCellSet

from .eos import TwoPhaseEOS
from .closure import Closure
from .state import Q, ConsFields, GradFields


class TwoPhaseProblem(Problem):
    """A class representing a two-phase system problem governed by the
    equations first described in :cite:`baer_two-phase_1986`"""

    def __init__(self, eos: TwoPhaseEOS, closure: Closure):
        self.eos = eos
        self.closure = closure

    def B(self, cells: Union[CellSet, MeshCellSet]):
        r""" This returns the tensor that pre-multiplies the non-conservative
        term of the problem.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull


        This method needs in general to return
        :math:`\pdeNonConservativeMultiplier` that for this case would be

        .. math::

            \pdeNonConservativeMultiplier_r =
            \begin{bmatrix}
            {u_r}_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            -p_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            -p_I {u_r}_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            p_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            p_I {u_r}_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            \end{bmatrix} \qquad r = 1 \dotso N_\text{dim}

        But since most of the :math:`\pdeNonConservativeMultiplier` is zero,
        since we just have the terms that pre-multiply
        :math:`\gradient{\alpha}` we just return :math:`B_{p1r}` that is:

        .. math::

            \tilde{\vb{B}}\qty(\pdeState) =
            \begin{bmatrix}
            u_I & v_I \\
            0 & 0 \\
            -p_I & 0 \\
            0 & -p_I \\
            -p_I u_I & -p_I v_I \\
            0 & 0 \\
            p_I & 0 \\
            0 & p_I \\
            p_I u_I & p_I v_I \\
            \end{bmatrix}

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` that contains the cell data

        eos
            An implementation of the equation of state. In particular it needs
            to implement the :class:`~.Closure` trait in order to be able to
            return `pI` and `uI` (in addition to the :class:`~.euler.EOS`)
        """
        values = cells.values.view(Q)

        num_cells_x, num_cells_y, num_fields = values.shape

        B = np.zeros(
            (
                num_cells_x,
                num_cells_y,
                num_fields,
                len(GradFields),
                MAX_DIMENSIONALITY,
            )
        )

        # Compute pI
        pI = self.closure.pI(values)

        # This is the vector (uI, vI)
        UI_VI = self.closure.uI(values)

        # First component of (uI, vI) along x
        UI = UI_VI[..., [Direction.X]]
        pIUI = np.multiply(pI, UI)

        # Second component of (uI, vI) along y
        VI = UI_VI[..., [Direction.Y]]
        pIVI = np.multiply(pI, VI)

        # Gradient component along x
        B[..., [Q.fields.alpha], GradFields.alpha, Direction.X] = UI
        B[..., [Q.fields.arhoU1], GradFields.alpha, Direction.X] = -pI
        B[..., [Q.fields.arhoE1], GradFields.alpha, Direction.X] = -pIUI
        B[..., [Q.fields.arhoU2], GradFields.alpha, Direction.X] = pI
        B[..., [Q.fields.arhoE2], GradFields.alpha, Direction.X] = pIUI

        # Gradient component along y
        B[..., [Q.fields.alpha], GradFields.alpha, Direction.Y] = VI
        B[..., [Q.fields.arhoV1], GradFields.alpha, Direction.Y] = -pI
        B[..., [Q.fields.arhoE1], GradFields.alpha, Direction.Y] = -pIVI
        B[..., [Q.fields.arhoV2], GradFields.alpha, Direction.Y] = pI
        B[..., [Q.fields.arhoE2], GradFields.alpha, Direction.Y] = pIVI

        return B

    def F(self, cells: Union[MeshCellSet, CellSet]) -> np.ndarray:
        r""" This returns the tensor representing the flux for a two-fluid model
        as described originally by :cite:`baer_two-phase_1986`


        Parameters
        ----------
        values
            A :class:`np.ndarray` that has dimension :math:`Nx \times Ny \times
            19` containing the values for all the values variables in all the
            mesh points

        Returns
        ---------
        F
            An array of dimension :math:`Nx \times Ny \times 9 \times 2`, i.e.
            an array that of each cell in :math:`x` and :math:`y` direction
            stores the :math:`9 \times 2` flux tensor

            The flux tensor is:

            .. math::
                \pdeConvective =
                \begin{bmatrix}
                    0 & 0 \\
                    \alpha_1 \rho u_1 & \alpha_1 \rho v_1 \\
                    \alpha_1(\rho_1 u_1^2 + p_1) & \alpha_1 \rho_1 u_1 v_1 \\
                    \alpha_1 \rho_1 v_1 u_1 & \alpha_1(\rho v_1^ 2 + p_1) \\
                    \alpha_1(\rho_1 E_1 + p_1)u_1 &
                        \alpha_1 (\rho_1 E + p)v_1 \\
                    \alpha_2 \rho u_2 & \alpha_2 \rho v_2 \\
                    \alpha_2(\rho_2 u_2^2 + p_2) & \alpha_2 \rho_2 u_2 v_2 \\
                    \alpha_2 \rho_2 v_2 u_2 & \alpha_2(\rho v_2^ 2 + p_2) \\
                    \alpha_2(\rho_2 E_2 + p_2)u_2 & \alpha_2 (\rho_2 E + p)v_2
                \end{bmatrix}
        """
        values: Q = cells.values.view(Q)
        num_cells_x, num_cells_y, _ = values.shape

        # Flux tensor
        F = np.zeros(
            (num_cells_x, num_cells_y, len(ConsFields), MAX_DIMENSIONALITY)
        )
        fields = values.fields

        alpha1 = values[..., fields.alpha]
        arhoU1 = values[..., fields.arhoU1]
        arhoV1 = values[..., fields.arhoV1]
        arhoE1 = values[..., fields.arhoE1]
        U1 = values[..., fields.U1]
        V1 = values[..., fields.V1]
        p1 = values[..., fields.p1]
        ap1 = alpha1 * p1

        alpha2 = 1 - alpha1
        arhoUU1 = np.multiply(arhoU1, U1)
        arhoUV1 = np.multiply(arhoU1, V1)
        arhoVV1 = np.multiply(arhoV1, V1)
        arhoVU1 = arhoUV1  # np.multiply(arhoV1, U1)

        arhoU2 = values[..., fields.arhoU2]
        arhoV2 = values[..., fields.arhoV2]
        arhoE2 = values[..., fields.arhoE2]
        U2 = values[..., fields.U2]
        V2 = values[..., fields.V2]
        p2 = values[..., fields.p2]
        ap2 = alpha2 * p2

        arhoUU2 = np.multiply(arhoU2, U2)
        arhoUV2 = np.multiply(arhoU2, V2)
        arhoVV2 = np.multiply(arhoV2, V2)
        arhoVU2 = arhoUV2  # np.multiply(arhoV2, U2)

        F[..., ConsFields.arho1, Direction.X] = arhoU1
        F[..., ConsFields.arho1, Direction.Y] = arhoV1
        F[..., ConsFields.arhoU1, Direction.X] = arhoUU1 + ap1
        F[..., ConsFields.arhoU1, Direction.Y] = arhoUV1
        F[..., ConsFields.arhoV1, Direction.X] = arhoVU1
        F[..., ConsFields.arhoV1, Direction.Y] = arhoVV1 + ap1
        F[..., ConsFields.arhoE1, Direction.X] = np.multiply(arhoE1 + ap1, U1)
        F[..., ConsFields.arhoE1, Direction.Y] = np.multiply(arhoE1 + ap1, V1)

        F[..., ConsFields.arho2, Direction.X] = arhoU2
        F[..., ConsFields.arho2, Direction.Y] = arhoV2
        F[..., ConsFields.arhoU2, Direction.X] = arhoUU2 + ap2
        F[..., ConsFields.arhoU2, Direction.Y] = arhoUV2
        F[..., ConsFields.arhoV2, Direction.X] = arhoVU2
        F[..., ConsFields.arhoV2, Direction.Y] = arhoVV2 + ap2
        F[..., ConsFields.arhoE2, Direction.X] = np.multiply(arhoE2 + ap2, U2)
        F[..., ConsFields.arhoE2, Direction.Y] = np.multiply(arhoE2 + ap2, V2)

        return F
