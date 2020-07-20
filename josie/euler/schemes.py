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


from josie.solver.scheme import ConvectiveScheme


from .eos import EOS
from .problem import EulerProblem
from .state import Q


class EulerScheme(ConvectiveScheme):
    """ A general base class for Euler schemes """

    def __init__(self, eos: EOS):
        self.problem: EulerProblem = EulerProblem(eos)

    def post_step(self, values: Q):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        fields = values.fields

        rho = values[..., fields.rho]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        rhoE = values[..., fields.rhoE]

        U = np.divide(rhoU, rho)
        V = np.divide(rhoV, rho)

        rhoe = rhoE - 0.5 * rho * (np.power(U, 2) + np.power(V, 2))
        e = np.divide(rhoe, rho)

        p = self.problem.eos.p(rho, e)
        c = self.problem.eos.sound_velocity(rho, p)

        values[..., fields.rhoe] = rhoe
        values[..., fields.U] = U
        values[..., fields.V] = V
        values[..., fields.p] = p
        values[..., fields.c] = c

    @staticmethod
    def compute_U_norm(values: Q, normals: np.ndarray):
        """Returns the value of the normal velocity component to the given
        ``normals``.

        Parameters
        ----------
        values
            A :class:`np.ndarray` that has dimension :math:`Nx \times Ny \times
            N_\text{fields}` containing the values for all the states in all
            the mesh points

        normals
            A :class:`np.ndarray` that has the dimensions :math:`Nx \times Ny
            \times N_\text{centroids} \times 2` containing the values of the
            normals to the faces of the cell

        Returns
        -------
        The value of the normal velocity
        """
        fields = values.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = values[..., np.newaxis, UV_slice]

        # Find the normal velocity
        U = np.einsum("...kl,...l->...k", UV, normals)

        return U

    def CFL(
        self,
        values: Q,
        volumes: np.ndarray,
        surfaces: np.ndarray,
        CFL_value,
    ) -> float:

        fields = values.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = values[..., UV_slice]

        U = np.linalg.norm(UV, axis=-1)
        c = values[..., fields.c]

        sigma = np.max(Rusanov.compute_sigma(U, c))

        # Min face surface
        # TODO: This probably needs to be generalized for 3D
        dx = np.min(volumes[..., np.newaxis] / surfaces)

        return CFL_value * dx / sigma


class HLL(EulerScheme):
    def F(
        self,
        values: Q,
        neigh_values: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        """This method implements the HLL scheme."""

        FS = np.zeros_like(values).view(Q)
        F = np.zeros_like(values.get_conservative())
        Q_L, Q_R = values, neigh_values
        fields = values.fields

        # Get normal velocities
        U_L = self.compute_U_norm(Q_L, normals)
        U_R = self.compute_U_norm(Q_R, normals)

        # Get sound speed
        a_L = Q_L[..., np.newaxis, fields.c]
        a_R = Q_R[..., np.newaxis, fields.c]

        # Compute the values of the wave velocities on every cell
        (sigma_L, sigma_R) = (U_L - a_L, U_R + a_R)

        F_L = np.einsum("...kl,...l->...k", self.problem.F(Q_L), normals)
        F_R = np.einsum("...kl,...l->...k", self.problem.F(Q_R), normals)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Qc_L = values.get_conservative()
        Qc_R = neigh_values.get_conservative()

        np.copyto(F, F_L, where=(sigma_L > 0))
        np.copyto(F, F_R, where=(sigma_R < 0))
        np.copyto(
            F,
            np.divide(
                sigma_R * F_L
                - sigma_L * F_R
                + sigma_L * sigma_R * (Qc_R - Qc_L),
                sigma_R - sigma_L,
            ),
            where=(sigma_L <= 0) * (sigma_R >= 0),
        )

        FS.set_conservative(surfaces[..., np.newaxis] * F)

        return FS


class Rusanov(EulerScheme):
    @staticmethod
    def compute_sigma(U_norm: np.ndarray, c: np.ndarray) -> np.ndarray:
        r"""Returns the value of the :math:`\sigma`(i.e. the wave velocity) for
        the the Rusanov scheme.

        .. math::

            \sigma = \max{\qty(\norm{\vb{u}} + c, \norm{\vb{u}} - c)}


        Parameters
        ----------
        U_norm
            The value of scalar velocity (generally is the normal velocity
            to the face) to use to compute the sigma

        c

            The value of sound velocity to use to compute the sigma

        Returns
        -------
        sigma
            A :math:`Nx \times Ny \times 1` containing the value of the sigma
            per each cell
        """

        sigma = np.abs(U_norm) + c

        # Add a dimension to have the right broadcasting
        sigma = sigma[..., np.newaxis]

        return sigma

    def F(
        self,
        values: Q,
        neigh_values: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):

        r"""This method implements the Rusanov scheme. See
        :cite:`toro_riemann_2009` for a detailed view on compressible schemes.
        The Rusanov scheme is discretized by:

        .. math::

            \numConvective  =
                \frac{1}{2} \qty[%
                \qty|\pdeConvective|_{i+1} + \qty|\pdeConvective|_{i}
                - \sigma \qty(\pdeState_{i+1} - \pdeState_{i})
                ] S_f
        """

        FS = np.zeros_like(values).view(Q)
        fields = values.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = values[..., UV_slice]
        UV_neigh = neigh_values[..., UV_slice]

        U = np.linalg.norm(UV, axis=-1)
        U_neigh = np.linalg.norm(UV_neigh, axis=-1)

        # Speed of sound
        c = values[..., fields.c]
        c_neigh = neigh_values[..., fields.c]

        # Let's retrieve the values of the sigma on every cell
        # for current cell
        sigma = self.compute_sigma(U, c)
        # and its neighbour
        sigma_neigh = self.compute_sigma(U_neigh, c_neigh)

        # Concatenate everything in a single array
        sigma_array = np.concatenate((sigma, sigma_neigh), axis=-1)

        # And the we found the max on the last axis (i.e. the maximum value
        # of sigma for each cell)
        sigma = np.max(sigma_array, axis=-1, keepdims=True)

        DeltaF = 0.5 * (self.problem.F(values) + self.problem.F(neigh_values))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...kl,...l->...k", DeltaF, normals)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        values_cons = values.get_conservative()
        neigh_values_cons = neigh_values.get_conservative()

        DeltaQ = 0.5 * sigma * (neigh_values_cons - values_cons)

        FS.set_conservative(surfaces[..., np.newaxis] * (DeltaF - DeltaQ))

        return FS
