# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from josie.state import State
from josie.euler.problem import EulerProblem
from josie.euler.state import EulerState
from josie.scheme.convective import ConvectiveScheme

from josie.general.schemes.space.muscl import MUSCL


if TYPE_CHECKING:
    from josie.euler.eos import EOS
    from josie.mesh.cellset import MeshCellSet


class EulerScheme(ConvectiveScheme):
    """A general base class for Euler schemes"""

    problem: EulerProblem

    def __init__(self, eos: EOS):
        super().__init__(EulerProblem(eos))

    def auxilliaryVariableUpdate(self, values: State):
        fields = EulerState.fields

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
        values[..., fields.e] = e

    def post_step(self, values: State):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        self.auxilliaryVariableUpdate(values)

    @staticmethod
    def compute_U_norm(values: State, normals: np.ndarray):
        r"""Returns the value of the normal velocity component to the given
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
        fields = EulerState.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = values[..., np.newaxis, UV_slice]

        # Find the normal velocity
        U = np.einsum("...mkl,...l->...mk", UV, normals)

        return U

    def CFL(self, cells: MeshCellSet, CFL_value: float) -> float:
        dt = super().CFL(cells, CFL_value)

        values: EulerState = cells.values.view(EulerState)
        fields = values.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = cells.values[..., UV_slice]

        U = np.linalg.norm(UV, axis=-1, keepdims=True)
        c = cells.values[..., fields.c]

        sigma = np.max(np.abs(U) + c[..., np.newaxis])

        # Min mesh dx
        dx = cells.min_length

        new_dt = CFL_value * dx / sigma

        return np.min((dt, new_dt))


class BerthonScheme(MUSCL):
    """An optional class to use the Berthon limiter in addition to the usual
    limiters for Euler equations. See in Berthon, Christophe. « Why the
    MUSCL–Hancock Scheme Is L1-Stable ». Numerische Mathematik, nᵒ 104 (2006):
    27‑46. https://doi.org/10.1007/s00211-006-0007-4."""

    @staticmethod
    def array_max_min(arr1: np.ndarray, arr2: np.ndarray, arr3: np.ndarray):
        return np.stack([arr1, np.stack([arr2, arr3]).min(axis=0)]).max(axis=0)

    @staticmethod
    def array_min(arr1: np.ndarray, arr2: np.ndarray):
        return np.stack([arr1, arr2]).min(axis=0)

    def pre_extrapolation(self, cells: MeshCellSet):
        for d in range(cells.dimensionality):
            ind_left = 2 * d
            ind_right = 2 * d + 1
            slope_R = self.slopes[..., ind_right]

            # Get the conservative fields (here in 1D only)
            fields = EulerState.fields
            rho = self.cells.values[..., fields.rho]
            U = self.cells.values[..., fields.U]
            rhoE = self.cells.values[..., fields.rhoE]

            # Density slope correction
            slope_R[..., fields.rho] = self.array_max_min(
                -0.5 * rho * np.sqrt((rhoE - rho * 0.5 * U * U) / rhoE),
                0.5 * rho * np.sqrt((rhoE - rho * 0.5 * U * U) / rhoE),
                0.5 * slope_R[..., fields.rho],
            )

            # Momentum slope correction
            slope_R[..., fields.rhoU] = self.array_max_min(
                0.5 * U * slope_R[..., fields.rho]
                - 0.5
                * np.sqrt(
                    2
                    * (rho - 4 * slope_R[..., fields.rho] ** 2 / rho)
                    * (rhoE - rho * 0.5 * U * U)
                ),
                0.5 * U * slope_R[..., fields.rho]
                + 0.5
                * np.sqrt(
                    2
                    * (rho - 4 * slope_R[..., fields.rho] ** 2 / rho)
                    * (rhoE - rho * 0.5 * U * U)
                ),
                0.5 * slope_R[..., fields.rhoU],
            )

            # Energy
            slope_R[..., fields.rhoE] = self.array_max_min(
                -0.5
                * (
                    rhoE
                    - (rho * U + 2 * slope_R[..., fields.rhoU]) ** 2
                    / (2 * (rho + 2 * slope_R[..., fields.rho]))
                ),
                0.5
                * (
                    rhoE
                    - (rho * U - 2 * slope_R[..., fields.rhoU]) ** 2
                    / (2 * (rho - 2 * slope_R[..., fields.rho]))
                ),
                0.5 * slope_R[..., fields.rhoE],
            )

            # Without limiters
            self.slopes[..., ind_right] = slope_R
            self.slopes[..., ind_left] = -slope_R
