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
import copy
import math
from typing import TYPE_CHECKING
from josie.pgd.problem import PGDProblem
from josie.pgd.state import PGDState
from josie.pgd.fields import PGDFields
from josie.scheme.convective import ConvectiveScheme


if TYPE_CHECKING:
    from josie.mesh.cellset import NeighboursCellSet, MeshCellSet
    from josie.fluid.state import SingleFluidState


class PGDScheme(ConvectiveScheme):
    """A general base class for PGD schemes"""

    problem: PGDProblem
    M_ref: np.ndarray
    eM_ref_tab: np.ndarray
    J: np.ndarray
    eJ: np.ndarray
    K_ref: np.ndarray
    U_min: float
    U_max: float
    eps: float

    def __init__(self):
        super().__init__(PGDProblem())

    def accumulate(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float):

        # Compute fluxes computed eventually by the other terms (diffusive,
        # nonconservative, source)
        # super().accumulate(cells, neighs, t)
        # Add conservative contribution
        self._fluxes += np.einsum(
            "...,...,ij,...jk->...ik",
            self.eJ[..., neighs.direction],
            self.J,
            self.eM_ref_tab[neighs.direction],
            self.F(cells, neighs),
        )

    def init_limiter(self, cells: MeshCellSet):
        self.U_min = np.amin(cells.values[..., PGDFields.U])
        self.U_max = np.amax(cells.values[..., PGDFields.U])
        self.eps = 1e-12

    def limiter(self, cells: MeshCellSet):
        nx, ny, num_dofs, num_fields = cells.values.shape
        uavg = np.zeros_like(cells.values)
        uavg[..., 0, :] = 0.25 * (
            cells.values[..., 0, :]
            + cells.values[..., 1, :]
            + cells.values[..., 2, :]
            + cells.values[..., 3, :]
        )
        uavg[..., 1, :] = uavg[..., 0, :]
        uavg[..., 2, :] = uavg[..., 0, :]
        uavg[..., 3, :] = uavg[..., 0, :]
        ucell = uavg[..., 0, :]
        umin = self.U_min
        umax = self.U_max
        rhomin = self.eps
        for i in range(nx):
            for j in range(ny):
                if i == 0:
                    umin = min(
                        np.amin(cells.values[i, j, :, PGDFields.U]),
                        np.amin(cells.values[i + 1, j, :, PGDFields.U]),
                    )
                    umax = max(
                        np.amax(cells.values[i, j, :, PGDFields.U]),
                        np.amax(cells.values[i + 1, j, :, PGDFields.U]),
                    )
                elif i == nx - 1:
                    umin = min(
                        np.amin(cells.values[i, j, :, PGDFields.U]),
                        np.amin(cells.values[i - 1, j, :, PGDFields.U]),
                    )
                    umax = max(
                        np.amax(cells.values[i, j, :, PGDFields.U]),
                        np.amax(cells.values[i - 1, j, :, PGDFields.U]),
                    )
                else:
                    umin = min(
                        np.amin(cells.values[i, j, :, PGDFields.U]),
                        np.amin(cells.values[i + 1, j, :, PGDFields.U]),
                        np.amin(cells.values[i - 1, j, :, PGDFields.U]),
                    )
                    umax = max(
                        np.amax(cells.values[i, j, :, PGDFields.U]),
                        np.amax(cells.values[i + 1, j, :, PGDFields.U]),
                        np.amax(cells.values[i - 1, j, :, PGDFields.U]),
                    )
                theta = np.ones(num_dofs)
                s_a_e = np.zeros((num_dofs, 3))
                s_a_e = copy.deepcopy(cells.values[i, j, :, 0:3])
                for k in range(num_dofs):
                    theta_rho = 1.0
                    theta_umax = 1.0
                    theta_umin = 1.0
                    if ucell[i, j, PGDFields.rho] < rhomin or (
                        math.fabs(
                            s_a_e[k, PGDFields.rhoU] - ucell[i, j, PGDFields.rhoU]
                        )
                        < rhomin
                    ):
                        theta_rho = 0.0
                        theta_umax = 0.0
                        theta_umin = 0.0
                    else:
                        if s_a_e[k, 0] < rhomin:
                            theta_rho = (rhomin - ucell[i, j, 0]) / (
                                s_a_e[k, 0] - ucell[i, j, 0]
                            )
                        if s_a_e[k, 0] * umax - s_a_e[k, 1] < 0.0:
                            theta_umax = (-ucell[i, j, 0] * umax + ucell[i, j, 1]) / (
                                (s_a_e[k, 0] - ucell[i, j, 0]) * umax
                                - s_a_e[k, 1]
                                + ucell[i, j, 1]
                            )

                        if s_a_e[k, 1] - s_a_e[k, 0] * umin < 0.0:
                            theta_umin = (-ucell[i, j, 1] + ucell[i, j, 0] * umin) / (
                                s_a_e[k, 1]
                                - ucell[i, j, 1]
                                - (s_a_e[k, 0] - ucell[i, j, 0]) * umin
                            )
                    theta[k] = max(
                        0.0, min(theta[k], theta_rho, theta_umax, theta_umin)
                    )

                theta_cell = np.amin(theta)
                cells.values[i, j, :, :] = (
                    theta_cell * (cells.values[i, j, :, :] - uavg[i, j, :, :])
                    + uavg[i, j, :, :]
                )

    def post_step(self, cells: MeshCellSet):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        values: PGDState = cells.values.view(PGDState)

        fields = values.fields

        rho = values[..., fields.rho]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]

        U = np.divide(rhoU, rho)
        V = np.divide(rhoV, rho)

        values[..., fields.U] = U
        values[..., fields.V] = V

    @staticmethod
    def compute_U_norm(values: SingleFluidState, normals: np.ndarray):
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
        fields = values.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = values[..., np.newaxis, UV_slice]

        # Find the normal velocity
        U = np.einsum("...kl,...l->...k", UV, normals)

        return U

    def CFL(self, cells: MeshCellSet, CFL_value: float) -> float:

        dt = super().CFL(cells, CFL_value)

        values: PGDState = cells.values.view(PGDState)
        fields = values.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = cells.values[..., UV_slice]

        U = np.linalg.norm(UV, axis=-1, keepdims=True)
        c = cells.values[..., fields.c]

        sigma = np.max(np.abs(U))

        # Min mesh dx
        dx = cells.min_length

        new_dt = CFL_value * dx / sigma

        return np.min((dt, new_dt))
