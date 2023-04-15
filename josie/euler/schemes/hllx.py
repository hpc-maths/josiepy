# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from typing import Tuple

from josie.math import Direction

from josie.euler.state import EulerState
from josie.state import State

from .scheme import EulerScheme


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
            ),
            axis=-1,
            keepdims=True,
        )

        sigma_R: np.ndarray = np.max(
            np.concatenate(
                (U_L + c_L[..., np.newaxis], U_R + c_R[..., np.newaxis]),
                axis=-1,
            ),
            axis=-1,
            keepdims=True,
        )

        return sigma_L, sigma_R

    def intercellFlux(
        self,
        Q_L: EulerState,
        Q_R: EulerState,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        FS = np.zeros_like(Q_L).view(EulerState)
        fields = EulerState.fields

        # Get normal velocities
        U_L = self.compute_U_norm(Q_L, normals)
        U_R = self.compute_U_norm(Q_R, normals)

        # Get sound speed
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]

        # Compute the values of the wave velocities on every cell
        sigma_L, sigma_R = self.compute_sigma(U_L, U_R, c_L, c_R)

        F_L = np.einsum("...mkl,...l->...mk", self.problem.F(Q_L), normals)

        F_R = np.einsum("...mkl,...l->...mk", self.problem.F(Q_R), normals)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Qc_L = Q_L.view(EulerState).get_conservative()
        Qc_R = Q_R.view(EulerState).get_conservative()

        F = np.zeros_like(F_L)

        np.copyto(F, F_L, where=(sigma_L > 0))
        np.copyto(F, F_R, where=(sigma_R < 0))
        np.copyto(
            F,
            np.divide(
                sigma_R * F_L - sigma_L * F_R + sigma_L * sigma_R * (Qc_R - Qc_L),
                sigma_R - sigma_L,
            ),
            where=(sigma_L <= 0) * (sigma_R >= 0),
        )

        FS.set_conservative(surfaces[..., np.newaxis, np.newaxis] * F)

        return FS


class HLLC(HLL):
    r"""This class implements the HLLC scheme. See
    :cite:`toro_riemann_2009` for a detailed view on compressible schemes.
    """

    def intercellFlux(
        self, Q_L: State, Q_R: State, normals: np.ndarray, surfaces: np.ndarray
    ):
        FS = np.zeros_like(Q_L).view(EulerState)
        F = np.zeros_like(Q_L.view(EulerState).get_conservative())
        fields = EulerState.fields

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
        U_L = np.einsum("...mkl,...l->...mk", UV_L, normals)
        U_R = np.einsum("...mkl,...l->...mk", UV_R, normals)

        # Speed of sound
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]

        # Compute the values of the wave velocities on every cell
        sigma_L, sigma_R = self.compute_sigma(U_L, U_R, c_L, c_R)

        # Compute the approximate contact discontinuity speed
        S_star = np.divide(
            p_R - p_L + rho_L * U_L * (sigma_L - U_L) - rho_R * U_R * (sigma_R - U_R),
            rho_L * (sigma_L - U_L) - rho_R * (sigma_R - U_R),
        )

        # This is the flux tensor dot the normal
        F_L = np.einsum("...mkl,...l->...mk", self.problem.F(Q_L), normals)
        F_R = np.einsum("...mkl,...l->...mk", self.problem.F(Q_R), normals)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Qc_L = Q_L.view(EulerState).get_conservative()
        Qc_R = Q_R.view(EulerState).get_conservative()

        # Init the intermediate states
        Q_star_R = np.empty_like(Qc_R)
        Q_star_L = np.empty_like(Qc_L)

        # FIXME: This can be avoided using direct flux expressions, Toro
        # p325, eq 10.41
        U_star_L = UV_L + np.einsum("...mk,...l->...mkl", (S_star - U_L), normals)
        U_star_R = UV_R + np.einsum("...mk,...l->...mkl", (S_star - U_R), normals)

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

        Q_star_R = Q_star_R * (rho_R * np.divide(sigma_R - U_R, sigma_R - S_star))
        Q_star_L = Q_star_L * (rho_L * np.divide(sigma_L - U_L, sigma_L - S_star))

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

        FS.set_conservative(
            (surfaces[..., np.newaxis, np.newaxis] * F).view(EulerState)
        )

        return FS
