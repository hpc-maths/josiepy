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
from josie.solver.euler.problem import EulerProblem
from josie.math import Direction

from .eos import EOS
from .closure import Closure
from .state import Q, Phases, PhasePair


class TwoPhaseProblem(Problem):
    """ A class representing a two-phase system problem governed by the
    equations first described in :cite:`baer_nunziato` """

    def __init__(self, eos: EOS, closure: Closure):
        # We re-use the EulerProblem code
        eos1 = eos[Phases.PHASE1]
        eos2 = eos[Phases.PHASE2]
        self._subproblems = PhasePair(EulerProblem(eos1), EulerProblem(eos2))

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

        num_fields = len(Q.fields)

        B = np.zeros((num_fields, num_fields, DIMENSIONALITY))

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
        B[Q.fields.alpha, Q.fields.alpha, Direction.X] = UI
        B[Q.fields.rhoU1, Q.fields.alpha, Direction.X] = -pI
        B[Q.fields.rhoE1, Q.fields.alpha, Direction.X] = -pIUI
        B[Q.fields.rhoU2, Q.fields.alpha, Direction.X] = pI
        B[Q.fields.rhoE2, Q.fields.alpha, Direction.X] = pIUI

        # Gradient component along y
        B[Q.fields.alpha, Q.fields.alpha, Direction.Y] = VI
        B[Q.fields.rhoV1, Q.fields.alpha, Direction.Y] = -pI
        B[Q.fields.rhoE1, Q.fields.alpha, Direction.Y] = -pIVI
        B[Q.fields.rhoV2, Q.fields.alpha, Direction.Y] = pI
        B[Q.fields.rhoE2, Q.fields.alpha, Direction.Y] = pIVI

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

        # Calculate the flux using the Euler flux per each phase
        # each phase state hase 9 fields
        for phase in Phases:
            F[..., phase : phase + 9, :] = self._subproblems[phase].F(
                state_array.get_phase(phase)
            )

        return F
