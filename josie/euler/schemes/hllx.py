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
from __future__ import annotations

import numpy as np

from typing import Tuple, TYPE_CHECKING

from josie.math import Direction
from josie.euler.state import Q

from .scheme import EulerScheme

if TYPE_CHECKING:
    from josie.mesh.cellset import CellSet, MeshCellSet


class HLL(EulerScheme):
    r"""This class implements the HLL scheme. See
    :cite:`toro_riemann_2009` for a detailed view on compressible schemes.
    """

    @staticmethod
    def compute_sigma(
        U_L: np.ndarray, U_R: np.ndarray, c_L: np.ndarray, c_R: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Returns the value of the :math:`\sigma`(i.e. the wave velocity) for
        the the HLL and HLLC scheme.

        .. math::

            \sigma_L = \min{\qty(\norm{\vb{u}_L} - c_L,
                \norm{\vb{u}_R} - c_R)}

            \sigma_R = \max{\qty(\norm{\vb{u}_L} + c_L,
                \norm{\vb{u}_R} + c_R)}


        Parameters
        ----------
        U_L
            The value of scalar velocity for each cell. Array dimensions
            :math:`N_x \times N_y \times 1`

        U_R
            The value of scalar velocity for each cell neighbour. Array
            dimensions :math:`N_x \times N_y \times 1`

        c_L
            The value of sound velocity for each cell

        c_R
            The value of sound velocity for each cell neighbour

        Returns
        -------
        sigma_L
            A :math:`Nx \times Ny \times 1` containing the value of the
            :math:`\sigma_L` per each cell

        sigma_R
            A :math:`Nx \times Ny \times 1` containing the value of the
            :math:`\sigma_R` per each cell
        """

        sigma_L: np.ndarray = np.min(
            np.concatenate(
                (U_L - c_L[..., np.newaxis], U_R - c_R[..., np.newaxis]),
                axis=-1,
            )
        )

        sigma_R: np.ndarray = np.max(
            np.concatenate(
                (U_L + c_L[..., np.newaxis], U_R + c_R[..., np.newaxis]),
                axis=-1,
            )
        )

        return sigma_L, sigma_R

    def F(self, cells: MeshCellSet, neighs: CellSet):

        values: Q = cells.values.view(Q)

        FS = np.zeros_like(values).view(Q)
        Q_L, Q_R = values, neighs.values.view(Q)
        fields = values.fields

        # Get normal velocities
        U_L = self.compute_U_norm(Q_L, neighs.normals)
        U_R = self.compute_U_norm(Q_R, neighs.normals)

        # Get sound speed
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]

        # Compute the values of the wave velocities on every cell
        sigma_L, sigma_R = self.compute_sigma(U_L, U_R, c_L, c_R)

        F_L = np.einsum(
            "...kl,...l->...k", self.problem.F(cells), neighs.normals
        )
        F_R = np.einsum(
            "...kl,...l->...k", self.problem.F(neighs), neighs.normals
        )

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Qc_L = Q_L.get_conservative()
        Qc_R = Q_R.get_conservative()

        F = np.zeros_like(F_L)

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

        FS.set_conservative(neighs.surfaces[..., np.newaxis] * F)

        return FS


class HLLC(HLL):
    r"""This class implements the HLLC scheme. See
    :cite:`toro_riemann_2009` for a detailed view on compressible schemes.
    """

    def F(self, cells: MeshCellSet, neighs: CellSet):

        values: Q = cells.values.view(Q)

        FS = np.zeros_like(values).view(Q)
        F = np.zeros_like(values.get_conservative())
        fields = Q.fields
        Q_L, Q_R = values, neighs.values.view(Q)

        # Get density
        rho_L = Q_L[..., np.newaxis, fields.rho]
        rho_R = Q_R[..., np.newaxis, fields.rho]

        # Get energy
        rhoE_L = Q_L[..., np.newaxis, fields.rhoE]
        rhoE_R = Q_R[..., np.newaxis, fields.rhoE]

        # Get pressure
        p_L = Q_L[..., np.newaxis, fields.p]
        p_R = Q_R[..., np.newaxis, fields.p]

        # Get velocity
        UV_slice = slice(fields.U, fields.V + 1)
        UV_L = Q_L[..., np.newaxis, UV_slice]
        UV_R = Q_R[..., np.newaxis, UV_slice]

        # Compute the normal velocity components
        U_L = np.einsum("...kl,...l->...k", UV_L, neighs.normals)
        U_R = np.einsum("...kl,...l->...k", UV_R, neighs.normals)

        # Speed of sound
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]

        # Compute the values of the wave velocities on every cell
        sigma_L, sigma_R = self.compute_sigma(U_L, U_R, c_L, c_R)

        # Compute the approximate contact discontinuity speed
        S_star = np.divide(
            p_R
            - p_L
            + rho_L * U_L * (sigma_L - U_L)
            - rho_R * U_R * (sigma_R - U_R),
            rho_L * (sigma_L - U_L) - rho_R * (sigma_R - U_R),
        )

        # This is the flux tensor dot the normal
        F_L = np.einsum(
            "...kl,...l->...k", self.problem.F(cells), neighs.normals
        )
        F_R = np.einsum(
            "...kl,...l->...k", self.problem.F(neighs), neighs.normals
        )

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Qc_L = Q_L.get_conservative()
        Qc_R = Q_R.get_conservative()

        # Init the intermediate states
        Q_star_R = np.empty_like(Qc_R)
        Q_star_L = np.empty_like(Qc_L)

        # FIXME: This can be avoided using direct flux expressions, Toro
        # p325, eq 10.41
        U_star_L = UV_L + np.einsum(
            "...k,...l->...kl", (S_star - U_L), neighs.normals
        )
        U_star_R = UV_R + np.einsum(
            "...k,...l->...kl", (S_star - U_R), neighs.normals
        )

        # Compute the intermediate states
        Q_star_R[..., fields.rho] = 1
        Q_star_L[..., fields.rho] = 1
        Q_star_R[..., fields.rhoU, np.newaxis] = U_star_R[..., Direction.X]
        Q_star_L[..., fields.rhoU, np.newaxis] = U_star_L[..., Direction.X]
        Q_star_R[..., fields.rhoV, np.newaxis] = U_star_R[..., Direction.Y]
        Q_star_L[..., fields.rhoV, np.newaxis] = U_star_L[..., Direction.Y]
        Q_star_R[..., fields.rhoE, np.newaxis] = np.divide(rhoE_R, rho_R) + (
            S_star - U_R
        ) * (S_star + np.divide(p_R, rho_R * (sigma_R - U_R)))
        Q_star_L[..., fields.rhoE, np.newaxis] = np.divide(rhoE_L, rho_L) + (
            S_star - U_L
        ) * (S_star + np.divide(p_L, rho_L * (sigma_L - U_L)))

        Q_star_R *= rho_R * np.divide(sigma_R - U_R, sigma_R - S_star)
        Q_star_L *= rho_L * np.divide(sigma_L - U_L, sigma_L - S_star)

        # Right supersonic flow
        np.copyto(F, F_L, where=(sigma_L >= 0))

        # Left supersonic flow
        np.copyto(F, F_R, where=(sigma_R < 0))

        # Subsonic flow - left state
        np.copyto(
            F,
            F_L + sigma_L * (Q_star_L - Qc_L),
            where=(sigma_L < 0) * (0 <= S_star),
        )

        # Subsonic flow - right state
        np.copyto(
            F,
            F_R + sigma_R * (Q_star_R - Qc_R),
            where=(S_star < 0) * (0 <= sigma_R),
        )

        FS.set_conservative(neighs.surfaces[..., np.newaxis] * F)

        return FS
