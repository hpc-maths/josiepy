# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from josie.euler.state import EulerState
from josie.state import State

from .scheme import EulerScheme


class Rusanov(EulerScheme):
    r"""This class implements the Rusanov scheme. See
    :cite:`toro_riemann_2009` for a detailed view on compressible schemes.
    The Rusanov scheme is discretized by:

    .. math::

        \numConvective  =
            \frac{1}{2} \qty[%
            \qty|\pdeConvective|_{i+1} + \qty|\pdeConvective|_{i}
            - \sigma \qty(\pdeState_{i+1} - \pdeState_{i})
            ] S_f
    """

    @staticmethod
    def compute_sigma(
        U_L: np.ndarray, U_R: np.ndarray, c_L: np.ndarray, c_R: np.ndarray
    ) -> np.ndarray:
        r"""Returns the value of the :math:`\sigma`(i.e. the wave velocity) for
        the the Rusanov scheme.

        .. math::

            \sigma = \max_{L, R}{\qty(\norm{\vb{u}} + c, \norm{\vb{u}} - c)}


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
        sigma
            A :math:`Nx \times Ny \times 1` containing the value of the sigma
            per each cell
        """

        sigma_L = np.abs(U_L) + c_L[..., np.newaxis]

        sigma_R = np.abs(U_R) + c_R[..., np.newaxis]

        # Concatenate everything in a single array
        sigma_array = np.concatenate((sigma_L, sigma_R), axis=-1)

        # And the we found the max on the last axis (i.e. the maximum value
        # of sigma for each cell)
        sigma = np.max(sigma_array, axis=-1, keepdims=True)

        return sigma

    def intercellFlux(
        self, Q_L: State, Q_R: State, normals: np.ndarray, surfaces: np.ndarray
    ):
        fields = EulerState.fields

        FS = np.zeros_like(Q_L).view(EulerState)

        # Get normal velocities
        U_L = self.compute_U_norm(Q_L, normals)
        U_R = self.compute_U_norm(Q_R, normals)

        # Speed of sound
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]

        sigma = self.compute_sigma(U_L, U_R, c_L, c_R)

        DeltaF = 0.5 * (self.problem.F(Q_L) + self.problem.F(Q_R))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...mkl,...l->...mk", DeltaF, normals)

        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Q_L_cons = Q_L.view(EulerState).get_conservative()
        Q_R_cons = Q_R.view(EulerState).get_conservative()

        DeltaQ = 0.5 * sigma * (Q_R_cons - Q_L_cons)

        FS.set_conservative(surfaces[..., np.newaxis, np.newaxis] * (DeltaF - DeltaQ))

        return FS
