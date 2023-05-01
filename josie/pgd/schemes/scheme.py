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
from typing import TYPE_CHECKING, List
from josie.pgd.problem import PGDProblem
from josie.pgd.state import Q
from josie.pgd.fields import PGDFields
from josie.scheme.convective import ConvectiveDGScheme


if TYPE_CHECKING:
    from josie.mesh.cellset import MeshCellSet
    from josie.fluid.state import SingleFluidState


class PGDScheme(ConvectiveDGScheme):
    """A general base class for PGD schemes"""

    problem: PGDProblem
    M_ref: np.ndarray
    eM_ref_tab: List
    J: np.ndarray
    eJ: np.ndarray
    K_ref: np.ndarray
    U_min: float
    U_max: float
    V_min: float
    V_max: float
    eps: float

    def __init__(self):
        super().__init__(PGDProblem())

    def post_integrate_fluxes(self, cells: MeshCellSet):
        super().post_integrate_fluxes(cells)
        self.limiter(cells)

    def stiffness_fluxes(self, cells: MeshCellSet) -> np.ndarray:
        vec = np.einsum(
            "ijk,...jlk->...ilk",
            self.K_ref,
            self.problem.F(cells),
        )

        vec = (2.0 / self.dx) * vec[..., 0] + (2.0 / self.dy) * vec[..., 1]
        vec2 = np.zeros(self._fluxes.shape)
        vec2.view(Q).set_conservative(vec)
        return vec2

    def init_limiter(self, cells: MeshCellSet):
        self.U_min = np.amin(cells.values[..., PGDFields.U])
        self.U_max = np.amax(cells.values[..., PGDFields.U])
        self.V_min = np.amin(cells.values[..., PGDFields.V])
        self.V_max = np.amax(cells.values[..., PGDFields.V])
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
        vmin = self.V_min
        vmax = self.V_max
        rhomin = self.eps

        for i in range(nx):
            for j in range(ny):
                theta = np.ones(num_dofs)
                s_a_e = np.zeros((num_dofs, 3))
                s_a_e = copy.deepcopy(cells.values[i, j, :, 0:3])
                for k in range(num_dofs):
                    theta_rho = 1.0
                    theta_umax = 1.0
                    theta_umin = 1.0
                    theta_vmax = 1.0
                    theta_vmin = 1.0
                    if s_a_e[k, PGDFields.rho] < rhomin:
                        theta_rho = (rhomin - ucell[i, j, PGDFields.rho]) / (
                            s_a_e[k, PGDFields.rho] - ucell[i, j, PGDFields.rho]
                        )
                    if s_a_e[k, PGDFields.rho] * vmax - s_a_e[k, PGDFields.rhoV] < 0.0:
                        theta_vmax = (
                            -ucell[i, j, PGDFields.rho] * vmax
                            + ucell[i, j, PGDFields.rhoV]
                        ) / (
                            (s_a_e[k, PGDFields.rho] - ucell[i, j, PGDFields.rho])
                            * vmax
                            - s_a_e[k, PGDFields.rhoV]
                            + ucell[i, j, PGDFields.rhoV]
                        )

                    if s_a_e[k, PGDFields.rhoV] - s_a_e[k, PGDFields.rho] * vmin < 0.0:
                        theta_vmin = (
                            -ucell[i, j, PGDFields.rhoV]
                            + ucell[i, j, PGDFields.rho] * vmin
                        ) / (
                            s_a_e[k, PGDFields.rhoV]
                            - ucell[i, j, PGDFields.rhoV]
                            - (s_a_e[k, PGDFields.rho] - ucell[i, j, PGDFields.rho])
                            * vmin
                        )
                    if s_a_e[k, PGDFields.rho] * umax - s_a_e[k, PGDFields.rhoU] < 0.0:
                        theta_umax = (
                            -ucell[i, j, PGDFields.rho] * umax
                            + ucell[i, j, PGDFields.rhoU]
                        ) / (
                            (s_a_e[k, PGDFields.rho] - ucell[i, j, PGDFields.rho])
                            * umax
                            - s_a_e[k, PGDFields.rhoU]
                            + ucell[i, j, PGDFields.rhoU]
                        )

                    if s_a_e[k, PGDFields.rhoU] - s_a_e[k, PGDFields.rho] * umin < 0.0:
                        theta_umin = (
                            -ucell[i, j, PGDFields.rhoU]
                            + ucell[i, j, PGDFields.rho] * umin
                        ) / (
                            s_a_e[k, PGDFields.rhoU]
                            - ucell[i, j, PGDFields.rhoU]
                            - (s_a_e[k, PGDFields.rho] - ucell[i, j, PGDFields.rho])
                            * umin
                        )
                    theta[k] = max(
                        0.0,
                        min(
                            theta[k],
                            theta_rho,
                            theta_umax,
                            theta_umin,
                            theta_vmax,
                            theta_vmin,
                        ),
                    )
                theta_cell = np.amin(theta)
                cells.values[i, j, :, :] = (
                    theta_cell * (cells.values[i, j, :, :] - uavg[i, j, :, :])
                    + uavg[i, j, :, :]
                )

    def post_step(self, values: Q):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        fields = Q.fields

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

        values: Q = cells.values.view(Q)
        fields = values.fields

        # Get the velocity components
        UV_slice = slice(fields.U, fields.V + 1)
        UV = cells.values[..., UV_slice]

        U = np.linalg.norm(UV, axis=-1, keepdims=True)

        sigma = np.max(np.abs(U))

        # Min mesh dx
        dx = cells.min_length

        new_dt = CFL_value * dx / sigma

        return np.min((dt, new_dt))
