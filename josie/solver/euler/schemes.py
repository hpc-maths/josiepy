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

from .problem import flux, eigs
from .state import Q


class Rusanov(Scheme):
    def convective_flux(
        self,
        values: Q,
        neigh_values: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        """ This schemes implements the Rusanov scheme. See :cite: `rusanov` for
        a detailed view on compressible schemes

        Parameters
        ----------
        values
            A :class:`Q` object that has dimension [Nx * Ny * 9] containing
            the values for all the states in all the mesh points
        neigh_values
            A :class:`Q` object  that has the same dimension of `values`. It
            contains the corresponding neighbour values of the states tored in
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
        FS = np.empty_like(values)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        values_cons = values.conservative()
        neigh_values_cons = neigh_values.conservative()

        # Let's retrieve the values of the eigenvalues
        eigs_values = eigs(values, normals)
        eigs_neigh_values = eigs(neigh_values, normals)

        sigma_array = np.concatenate((eigs_values, eigs_neigh_values), axis=-1)

        # And the we found the max on the last axis (i.e. the maximul value
        # of sigma for each cell)
        sigma = np.max(sigma_array, axis=-1)

        DeltaF = 0.5 * (flux(values) + flux(neigh_values))
        # This is the flux tensor dot the normal
        DeltaF = np.einsum("ijkl,ijl->ijk", DeltaF, normals)

        DeltaQ = (
            0.5 * sigma[:, :, np.newaxis] * (neigh_values_cons - values_cons)
        )

        FS[:, :, :4] = surfaces[:, :, np.newaxis] * (DeltaF - DeltaQ)

        return FS

    def CFL(
        self,
        values: np.ndarray,
        volumes: np.ndarray,
        normals: np.ndarray,
        surfaces: np.ndarray,
        CFL_value,
    ) -> float:
        UV = values[:, :, 5:7]
        c = values[:, :, -1]

        # TODO: We can probably optimize this since we compute `sigma` in
        # the rusanov scheme, so we could find a way to store it and avoid
        # to recompute it here

        # Absolute value squared for each cell
        # Equivalent to: U[:, :]**2 + V[:, :]**2
        UU_abs = np.einsum("ijk,ijk->ij", UV, UV)

        # Max speed value over all cells
        U_abs = np.sqrt(np.max(UU_abs))

        # Max sound velocity
        c_max = np.max(c)

        # Min face surface
        # TODO: This probably needs to be generalized for 3D
        dx = np.min(surfaces)

        return CFL_value * dx / (U_abs + c_max)
