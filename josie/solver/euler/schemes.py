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
        self.problem = EulerProblem(eos)

    def post_step(self, values: Q):
        """ During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        fields = Q.fields

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


class Rusanov(EulerScheme):
    def sigma(self, state_array: Q, normals: np.ndarray) -> np.ndarray:
        r""" Returns the value of the :math:`\sigma`(i.e. the wave velocity) for
        the the Rusanov scheme.

        .. math:

            \qty|\pdeConvective|_{i+\frac{1}{2}} =
                \frac{1}{2} \qty[%
                \qty|\pdeConvective|_{i+1} + \qty|\pdeConvective|_{i}
                - \sigma \qty(\pdeState_{i+1} - \pdeState{i})
                ]

        Parameters
        ----------
        state_array
            A :class:`Q` object that has dimension :math:`Nx \times Ny \times
            9` containing the values for all the states in all the mesh points

        normals
            A :class:`np.ndarray` that has the dimensions :math:`Nx \times Ny
            \times 2` containing the values of the normals to the face
            connecting the cell to its neighbour

        Returns
        -------
        sigma
            A :math:`Nx \times Ny \times 1` containing the value of the sigma
            per each cell
        """
        fields = Q.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = state_array[..., UV_slice]

        # Find the normal velocity
        # 2D: U = np.einsum("ijk,ijk->ij", UV, normals)
        U = np.einsum("...k,...k->...", UV, normals)

        # Speed of sound
        c = state_array[..., fields.c]

        sigma = np.abs(U) + c

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

        """ This method implements the Rusanov scheme. See :cite: `rusanov` for
        a detailed view on compressible schemes, given a suitable
        """

        FS = np.empty_like(values).view(Q)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        values_cons = values.get_conservative()
        neigh_values_cons = neigh_values.get_conservative()

        # Let's retrieve the values of the sigma on every cell
        # for current cell
        sigma = self.sigma(values, normals)
        # and its neighbour
        sigma_neigh = self.sigma(neigh_values, normals)

        # Concatenate everything in a single array
        sigma_array = np.concatenate((sigma, sigma_neigh), axis=-1)

        # And the we found the max on the last axis (i.e. the maximum value
        # of sigma for each cell)
        sigma = np.max(sigma_array, axis=-1)

        DeltaF = 0.5 * (self.problem.F(values) + self.problem.F(neigh_values))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...kl,...l->...k", DeltaF, normals)

        DeltaQ = (
            0.5 * sigma[..., np.newaxis] * (neigh_values_cons - values_cons)
        )

        FS.set_conservative(surfaces[..., np.newaxis] * (DeltaF - DeltaQ))

        return FS

    def CFL(
        self,
        values: Q,
        volumes: np.ndarray,
        normals: np.ndarray,
        surfaces: np.ndarray,
        CFL_value,
    ) -> float:
        UV = values[..., Q.fields.U : Q.fields.V + 1]
        c = values[..., Q.fields.c]

        # TODO: We can probably optimize this since we compute `sigma` in
        # the rusanov scheme, so we could find a way to store it and avoid
        # to recompute it here

        # Absolute value squared for each cell
        # Equivalent to: U[:, :]**2 + V[:, :]**2
        UU_abs = np.einsum("...k,...k->...", UV, UV)

        # Max speed value over all cells
        U_abs = np.sqrt(np.max(UU_abs))

        # Max sound velocity
        c_max = np.max(c)

        # Min face surface
        # TODO: This probably needs to be generalized for 3D
        dx = np.min(surfaces)

        return CFL_value * dx / (U_abs + c_max)
