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

from josie.solver.scheme import Scheme
from josie.solver.euler.schemes import Rusanov as EulerRusanov
from josie.solver.scheme.nonconservative import NonConservativeScheme
from josie.solver.scheme.convective import ConvectiveScheme

from .closure import Closure
from .eos import TwoPhaseEOS
from .problem import TwoPhaseProblem
from .state import Q, Phases, PhasePair


class TwoPhaseScheme(Scheme):
    """ A base class for a twophase scheme """

    def __init__(self, eos: TwoPhaseEOS, closure: Closure):
        self.problem: TwoPhaseProblem = TwoPhaseProblem(eos, closure)

    def post_step(self, values: Q):
        """ During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """
        alpha = values[..., values.fields.alpha]

        alphas = PhasePair(alpha, 1 - alpha)

        for phase in Phases:
            phase_values = values.get_phase(phase)
            alpha = alphas[phase]
            fields = phase_values.fields

            rho = phase_values[..., fields.arho] / alpha
            rhoU = phase_values[..., fields.arhoU] / alpha
            rhoV = phase_values[..., fields.arhoV] / alpha
            rhoE = phase_values[..., fields.arhoE] / alpha

            # idx = np.where(alpha <= 1e-1)

            # # Zero out fields with alpha too small
            # rho[idx] = 1e-1
            # rhoU[idx] = 1e-1
            # rhoV[idx] = 1e-1
            # rhoE[idx] = 1e-1

            # idx = np.where(alpha > 1e-1)

            # # The rest we can safely divide by alpha
            # rho[idx] /= alpha[idx]
            # rhoU[idx] /= alpha[idx]
            # rhoV[idx] /= alpha[idx]
            # rhoE[idx] /= alpha[idx]

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


class Upwind(NonConservativeScheme, TwoPhaseScheme):
    def accumulate(
        self,
        values: Q,
        neigh_values: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> Q:

        # Compute fluxes computed eventually by the other schemes (e.g.
        # conservative)
        fluxes = super().accumulate(values, neigh_values, normals, surfaces)

        # Add nonconservative contribution
        B = self.problem.B(values)
        G = self.G(values, neigh_values, normals, surfaces)
        fluxes += np.einsum("...ij,...j->...i", B, G)

        return fluxes

    def G(
        self,
        values: Q,
        neigh_values: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> np.ndarray:

        # nx, ny, _ = values.shape

        # alpha_face = np.zeros((nx, ny))

        # # Get vector of uI
        # UI_VI = self.problem.closure.uI(values)
        # UI_VI_neigh = self.problem.closure.uI(values)

        # # Normal uI
        # UI = np.einsum("...k,...k->...", UI_VI, normals)
        # UI_neigh = np.einsum("...k,...k->...", UI_VI_neigh, normals)

        # UI_face = 0.5 * (UI + UI_neigh)

        alpha = values[..., Q.fields.alpha]
        alpha_neigh = neigh_values[..., Q.fields.alpha]

        # # Upwind
        # # Cells where the normal interfacial velocity is > 0
        # idx = np.where(UI_face >= 0)

        # alpha_face[idx] = alpha_neigh[idx]

        # # Cells where the normal interfacial velocity is < 0
        # idx = np.where(UI_face < 0)
        # alpha_face[idx] = alpha[idx]

        alpha_face = 0.5 * (alpha + alpha_neigh)

        alphan = np.einsum("...i,...ij->...ij", alpha_face, normals)

        return surfaces[..., np.newaxis] * alphan


class Rusanov(ConvectiveScheme, TwoPhaseScheme):
    def F(
        self,
        values: Q,
        neigh_values: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> Q:
        r""" This schemes implements the Rusanov scheme for a
        :class:`~.TwoPhaseProblem`. It applies the :class:`~.euler.Rusanov`
        scheme indipendently for each phase (with the :math:`\sigma` correctly
        calculated among all the two phases state)

        Parameters
        ----------
        values
            A :class:`np.ndarray` that has dimension [Nx * Ny * 19] containing
            the values for all the states in all the mesh points
        neigh_values
            A :class:`np.ndarray` that has the same dimension of `values`. It
            contains the corresponding neighbour values of the state stored in
            `values`, i.e. the neighbour of `values[i]` is `neigh_values[i]`
        normals
            A :class:`np.ndarray` that has the dimensions [Nx * Ny * 2]
            containing the values of the normals to the face connecting the
            cell to its neighbour
        surfaces
            A :class:`np.ndarray` that has the dimensions [Nx * Ny] containing
            the values of the face surfaces of the face connecting the cell to
            is neighbour
        """
        FS = np.zeros_like(values).view(Q)

        # Compute the sigma per each phase
        sigmas = []

        alpha = values[..., values.fields.alpha]

        alphas = PhasePair(alpha, 1 - alpha)

        for phase in Phases:
            phase_values = values.get_phase(phase)
            phase_neigh_values = neigh_values.get_phase(phase)

            fields = phase_values.fields

            alpha = alphas[phase]

            # Get the velocity components
            UV_slice = slice(fields.U, fields.V + 1)
            UV = phase_values[..., UV_slice]
            UV_neigh = phase_neigh_values[..., UV_slice]

            U = np.linalg.norm(UV, axis=-1)
            U_neigh = np.linalg.norm(UV_neigh, axis=-1)

            # Speed of sound
            c = phase_values[..., fields.c]
            c_neigh = phase_neigh_values[..., fields.c]

            # Let's retrieve the values of the sigma on every cell
            # for current cell
            sigma = EulerRusanov.compute_sigma(U, c)
            # and its neighbour
            sigma_neigh = EulerRusanov.compute_sigma(U_neigh, c_neigh)

            # Concatenate everything in a single array
            sigma_array = np.concatenate((sigma, sigma_neigh), axis=-1)

            # And the we found the max on the last axis (i.e. the maximum value
            # of sigma for each cell)
            sigmas.append(np.max(sigma_array, axis=-1, keepdims=True))

        # Concatenate the sigmas for both phases in a single array
        sigma_array = np.concatenate(sigmas, axis=-1)

        # And the we found the max on the last axis (i.e. the maximum value
        # of sigma for each cell)
        sigma = np.max(sigma_array, axis=-1, keepdims=True)

        DeltaF = 0.5 * (self.problem.F(values) + self.problem.F(neigh_values))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...kl,...l->...k", DeltaF, normals)

        # Remove useless row related to alpha that has no convective flux
        DeltaF = DeltaF[..., 1:]

        values_cons = values.get_conservative()
        neigh_values_cons = neigh_values.get_conservative()

        DeltaQ = 0.5 * sigma * (neigh_values_cons - values_cons)

        # Remove first row related to alpha
        FS.set_conservative(surfaces[..., np.newaxis] * (DeltaF - DeltaQ))

        return FS

    def CFL(
        self, values: Q, volumes: np.ndarray, surfaces: np.ndarray, CFL_value,
    ) -> float:

        dt = 1e9
        dx = np.min(volumes[..., np.newaxis] / surfaces)
        alpha = values[..., Q.fields.alpha]
        alphas = PhasePair(alpha, 1 - alpha)
        for phase in Phases:
            phase_values = values.get_phase(phase)
            fields = phase_values.fields
            alpha = alphas[phase]

            # Get the velocity components
            UV_slice = slice(fields.U, fields.V + 1)
            UV = phase_values[..., UV_slice]

            U = np.linalg.norm(UV, axis=-1)
            c = phase_values[..., fields.c]

            sigma = np.max(EulerRusanov.compute_sigma(U, c))

            dt = np.min((dt, CFL_value * dx / sigma))

        return dt
