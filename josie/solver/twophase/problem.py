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


from josie.solver.problem import Problem
from josie.math import Direction

from .eos import TwoPhaseEOS
from .closure import Closure
from .state import Q


class TwoPhaseProblem(Problem):
    """ A class representing a two-phase system problem governed by the
    equations first described in :cite:`baer_nunziato` """

    def __init__(self, eos: TwoPhaseEOS, closure: Closure):
        self.eos = eos
        self.closure = closure

    def B(self, state_array: Q):
        r""" This returns the tensor that pre-multiplies the non-conservative
        term of the problem.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull


        This method needs to return :math:`\pdeNonConservativeMultiplier`

        .. math::

            \pdeNonConservativeMultiplier_x =
            \begin{bmatrix}
            u_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            -p_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            -p_I u_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            p_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            p_I u_I & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            \end{bmatrix}

        Parameters
        ----------
        Q
            The :class:`~.Q` array containing the values of all the fields

        eos
            An implementation of the equation of state. In particular it needs
            to implement the :class:`~.Closure` trait in order to be able to
            return `pI` and `uI` (in addition to the :class:`~.euler.EOS`)
        """
        # TODO: Needs to be generalized for 3D
        DIMENSIONALITY = 2

        num_cells_x, num_cells_y, num_fields = state_array.shape

        B = np.zeros(
            (num_cells_x, num_cells_y, num_fields, num_fields, DIMENSIONALITY)
        )

        # Compute pI
        pI = self.closure.pI(state_array)

        # This is the vector (uI, vI)
        UI_VI = self.closure.uI(state_array)

        # First component of (uI, vI) along x
        UI = UI_VI[..., Direction.X]
        pIUI = np.multiply(pI, UI)

        # Second component of (uI, vI) along y
        VI = UI_VI[..., Direction.Y]
        pIVI = np.multiply(pI, VI)

        # Gradient component along x
        B[..., Q.fields.alpha, Q.fields.alpha, Direction.X] = UI
        B[..., Q.fields.arhoU1, Q.fields.alpha, Direction.X] = -pI
        B[..., Q.fields.arhoE1, Q.fields.alpha, Direction.X] = -pIUI
        B[..., Q.fields.arhoU2, Q.fields.alpha, Direction.X] = pI
        B[..., Q.fields.arhoE2, Q.fields.alpha, Direction.X] = pIUI

        # Gradient component along y
        B[..., Q.fields.alpha, Q.fields.alpha, Direction.Y] = VI
        B[..., Q.fields.arhoV1, Q.fields.alpha, Direction.Y] = -pI
        B[..., Q.fields.arhoE1, Q.fields.alpha, Direction.Y] = -pIVI
        B[..., Q.fields.arhoV2, Q.fields.alpha, Direction.Y] = pI
        B[..., Q.fields.arhoE2, Q.fields.alpha, Direction.Y] = pIVI

        return B

    def F(self, state_array: Q) -> np.ndarray:
        r""" This returns the tensor representing the flux for a two-fluid model
        as described originally by :cite:`baer_nunziato`


        Parameters
        ----------
        state_array
            A :class:`np.ndarray` that has dimension :math:`Nx \times Ny \times
            19` containing the values for all the state variables in all the
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

        num_cells_x, num_cells_y, _ = state_array.shape

        # Flux tensor
        F = np.zeros((num_cells_x, num_cells_y, 9, 2))

        arhoU1 = state_array[..., Q.fields.arhoU1]
        arhoV1 = state_array[..., Q.fields.arhoV1]
        arhoE1 = state_array[..., Q.fields.arhoE1]
        U1 = state_array[..., Q.fields.U1]
        V1 = state_array[..., Q.fields.V1]
        p1 = state_array[..., Q.fields.p1]

        arhoUU1 = np.multiply(arhoU1, U1)
        arhoUV1 = np.multiply(arhoU1, V1)
        arhoVV1 = np.multiply(arhoV1, V1)
        arhoVU1 = np.multiply(arhoV1, U1)

        arhoU2 = state_array[..., Q.fields.arhoU2]
        arhoV2 = state_array[..., Q.fields.arhoV2]
        arhoE2 = state_array[..., Q.fields.arhoE2]
        U2 = state_array[..., Q.fields.U2]
        V2 = state_array[..., Q.fields.V2]
        p2 = state_array[..., Q.fields.p2]

        arhoUU2 = np.multiply(arhoU2, U2)
        arhoUV2 = np.multiply(arhoU2, V2)
        arhoVV2 = np.multiply(arhoV2, V2)
        arhoVU2 = np.multiply(arhoV2, U2)

        # First row F[..., 0, k] is related to alpha, that has no conservative
        # flux
        F[..., 1, 0] = arhoU1
        F[..., 1, 1] = arhoV1
        F[..., 2, 0] = arhoUU1 + p1
        F[..., 2, 1] = arhoUV1
        F[..., 3, 0] = arhoVU1
        F[..., 3, 1] = arhoVV1 + p1
        F[..., 4, 0] = np.multiply(arhoE1 + p1, U1)
        F[..., 4, 1] = np.multiply(arhoE1 + p1, V1)

        F[..., 5, 0] = arhoU2
        F[..., 5, 1] = arhoV2
        F[..., 6, 0] = arhoUU2 + p2
        F[..., 6, 1] = arhoUV2
        F[..., 7, 0] = arhoVU2
        F[..., 7, 1] = arhoVV2 + p2
        F[..., 8, 0] = np.multiply(arhoE2 + p2, U2)
        F[..., 8, 1] = np.multiply(arhoE2 + p2, V2)

        return F
