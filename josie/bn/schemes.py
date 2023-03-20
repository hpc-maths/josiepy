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

import numpy as np

from josie.mesh.cellset import NeighboursCellSet, MeshCellSet
from josie.euler.schemes import Rusanov as EulerRusanov
from josie.scheme import Scheme
from josie.scheme.nonconservative import NonConservativeScheme
from josie.scheme.convective import ConvectiveScheme
from josie.twofluid.state import PhasePair
from josie.twofluid.fields import Phases

from .closure import Closure
from .eos import TwoPhaseEOS
from .problem import TwoPhaseProblem
from .state import Q


class BaerScheme(Scheme):
    """A base class for a twophase scheme"""

    problem: TwoPhaseProblem

    def __init__(self, eos: TwoPhaseEOS, closure: Closure):
        super().__init__(TwoPhaseProblem(eos, closure))

    def auxilliaryVariableUpdate(self, values):
        alpha = values[..., Q.fields.alpha]

        alphas = PhasePair(alpha, 1 - alpha)

        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)
            alpha = alphas[phase]
            fields = phase_values.fields

            rho = phase_values[..., fields.arho] / alpha
            rhoU = phase_values[..., fields.arhoU] / alpha
            rhoV = phase_values[..., fields.arhoV] / alpha
            rhoE = phase_values[..., fields.arhoE] / alpha

            U = np.divide(rhoU, rho)
            V = np.divide(rhoV, rho)

            rhoe = rhoE - 0.5 * rho * (np.power(U, 2) + np.power(V, 2))
            e = np.divide(rhoe, rho)

            p = self.problem.eos[phase].p(rho, e)
            c = self.problem.eos[phase].sound_velocity(rho, p)

            phase_values[..., fields.rhoe] = rhoe
            phase_values[..., fields.U] = U
            phase_values[..., fields.V] = V
            phase_values[..., fields.p] = p
            phase_values[..., fields.c] = c

            values.view(Q).set_phase(
                phase,
                phase_values,
            )

    def post_step(self, values: Q):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        self.auxilliaryVariableUpdate(values)


class Upwind(BaerScheme, NonConservativeScheme):
    r"""An optimized upwind scheme that reduces the size of the
    :math:`\pdeNonConservativeMultiplier` knowing that for
    :cite:`baer_two-phase_1986` the only state variable appearing in the non
    conservative term is :math:`\alpha`. It concentratres the numerical
    flux computation into :meth:`G`.

    Check also :class:`~twofluid.problem.TwoPhaseProblem.B`.
    """

    def G(self, cells: MeshCellSet, neighs: NeighboursCellSet) -> np.ndarray:
        Q_L: Q = cells.values.view(Q)
        Q_R: Q = neighs.values.view(Q)

        nx, ny, num_dofs, _ = Q_L.shape

        alpha_face = np.zeros((nx, ny, num_dofs, 1))

        # Get vector of uI
        UI_VI_L = self.problem.closure.uI(Q_L)
        UI_VI_R = self.problem.closure.uI(Q_R)

        UI_VI_face = 0.5 * (UI_VI_L + UI_VI_R)

        # Normal uI
        U_face = np.einsum("...kl,...l->...k", UI_VI_face, neighs.normals)[
            ..., np.newaxis
        ]

        alpha_L = Q_L[..., [Q.fields.alpha]]
        alpha_R = Q_R[..., [Q.fields.alpha]]

        np.copyto(alpha_face, alpha_L, where=U_face >= 0)
        np.copyto(alpha_face, alpha_R, where=U_face < 0)

        alphan_face = np.einsum(
            "...mk,...l->...mkl", alpha_face, neighs.normals
        )

        # surfaces need to be broadcasted to comply with the alphan structure
        G = (
            neighs.surfaces[..., np.newaxis, np.newaxis, np.newaxis]
            * alphan_face
        )

        return G


class Rusanov(BaerScheme, ConvectiveScheme):
    def intercellFlux(
        self, Q_L: Q, Q_R: Q, normals: np.ndarray, surfaces: np.ndarray
    ) -> Q:
        r"""This schemes implements the Rusanov scheme for a
        :class:`TwoPhaseProblem`. It applies the :class:`~.euler.Rusanov`
        scheme indipendently for each phase (with the :math:`\sigma` correctly
        calculated among all the two phases state)

        Parameters
        ----------
        Q_L:
            State values in the "left" side of the interface

        Q_R:
            State values in the "right" side of the interface

        Returns
        -------
        F
            The value of the numerical convective flux multiplied by the
            surface value :math:`\numConvective`
        """

        FS = np.zeros_like(Q_L).view(Q)

        # Compute the sigma per each phase
        sigmas = []

        alpha = Q_L[..., Q.fields.alpha]

        alphas = PhasePair(alpha, 1 - alpha)

        for phase in Phases:
            phase_L = Q_L.view(Q).get_phase(phase)
            phase_R = Q_R.view(Q).get_phase(phase)

            fields = phase_L.fields

            alpha = alphas[phase]

            # Get normal velocities
            U_L = EulerRusanov.compute_U_norm(phase_L, normals)
            U_R = EulerRusanov.compute_U_norm(phase_R, normals)

            # Speed of sound
            c_L = phase_L[..., fields.c]
            c_R = phase_R[..., fields.c]

            # Let's retrieve the values of the sigma on every cell
            # for current cell
            sigma = EulerRusanov.compute_sigma(U_L, U_R, c_L, c_R)

            # And the we found the max on the last axis (i.e. the maximum value
            # of sigma for each cell)
            sigmas.append(sigma)

        # Concatenate the sigmas for both phases in a single array
        sigma_array = np.concatenate(sigmas, axis=-1)

        # And the we found the max on the last axis (i.e. the maximum value
        # of sigma for each cell)
        sigma = np.max(sigma_array, axis=-1, keepdims=True)

        DeltaF = 0.5 * (self.problem.F(Q_L) + self.problem.F(Q_R))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...mkl,...l->...mk", DeltaF, normals)

        Qc_L = Q_L.view(Q).get_conservative()
        Qc_R = Q_R.view(Q).get_conservative()

        DeltaQ = 0.5 * sigma * (Qc_R - Qc_L)

        FS.set_conservative(
            surfaces[..., np.newaxis, np.newaxis] * (DeltaF - DeltaQ)
        )

        return FS

    def CFL(
        self,
        cells: MeshCellSet,
        CFL_value,
    ) -> float:
        dt = super().CFL(cells, CFL_value)

        dx = cells.min_length
        alpha = cells.values[..., Q.fields.alpha]
        alphas = PhasePair(alpha, 1 - alpha)
        for phase in Phases:
            phase_values = cells.values.view(Q).get_phase(phase)
            fields = phase_values.fields
            alpha = alphas[phase]

            # Get the velocity components
            UV_slice = slice(fields.U, fields.V + 1)
            UV = phase_values[..., UV_slice]

            U = np.linalg.norm(UV, axis=-1)
            c = phase_values[..., fields.c]

            sigma = np.max(np.abs(U) + c[..., np.newaxis])

            dt = np.min((dt, CFL_value * dx / sigma))

        return dt
