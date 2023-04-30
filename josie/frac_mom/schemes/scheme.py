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
from josie.frac_mom.problem import FracMomProblem
from josie.frac_mom.state import Q
from josie.frac_mom.fields import FracMomFields
from josie.mesh.cellset import MeshCellSet
from josie.scheme.convective import ConvectiveDGScheme


if TYPE_CHECKING:
    from josie.fluid.state import SingleFluidState


class FracMomScheme(ConvectiveDGScheme):
    """A general base class for PGD schemes"""

    problem: FracMomProblem
    M_ref: np.ndarray
    eM_ref_tab: np.ndarray
    J: np.ndarray
    eJ: np.ndarray
    K_ref: np.ndarray
    U_min: float
    U_max: float
    V_min: float
    V_max: float
    eps: float

    def __init__(self):
        super().__init__(FracMomProblem())

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
        self.U_min = np.amin(cells.values[..., FracMomFields.U])
        self.U_max = np.amax(cells.values[..., FracMomFields.U])
        self.V_min = np.amin(cells.values[..., FracMomFields.V])
        self.V_max = np.amax(cells.values[..., FracMomFields.V])
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
        m0min = self.eps

        for i in range(nx):
            for j in range(ny):
                theta = np.ones(num_dofs)
                s_a_e = np.zeros((num_dofs, 6))
                s_a_e = copy.deepcopy(cells.values[i, j, :, 0:6])
                for k in range(num_dofs):
                    theta_m0 = 1.0
                    theta_m12a = 1.0
                    theta_m12b = 1.0
                    theta_m1a = 1.0
                    theta_m1b = 1.0
                    theta_m32a = 1.0
                    theta_m32b = 1.0
                    theta_umax = 1.0
                    theta_umin = 1.0
                    theta_vmax = 1.0
                    theta_vmin = 1.0
                    if (
                        ucell[i, j, FracMomFields.m0] < m0min
                        or (
                            math.fabs(
                                s_a_e[k, FracMomFields.m1U]
                                - ucell[i, j, FracMomFields.m1U]
                            )
                            < m0min
                        )
                        or (
                            math.fabs(
                                s_a_e[k, FracMomFields.m1V]
                                - ucell[i, j, FracMomFields.m1V]
                            )
                            < m0min
                        )
                        or (
                            ucell[i, j, FracMomFields.m0]
                            - ucell[i, j, FracMomFields.m1]
                            < m0min
                        )
                    ):
                        theta_m0 = 0.0
                        theta_m12a = 0.0
                        theta_m12b = 0.0
                        theta_m1a = 0.0
                        theta_m1b = 0.0
                        theta_m32a = 0.0
                        theta_m32b = 0.0
                        theta_umax = 0.0
                        theta_umin = 0.0
                        theta_vmax = 0.0
                        theta_vmin = 0.0
                    else:
                        H1a = s_a_e[k, FracMomFields.m12]
                        H1b = (
                            s_a_e[k, FracMomFields.m0]
                            - s_a_e[k, FracMomFields.m12]
                        )
                        H2a = (
                            s_a_e[k, FracMomFields.m0]
                            * s_a_e[k, FracMomFields.m1]
                            - s_a_e[k, FracMomFields.m12] ** 2
                        )
                        H2b = (
                            s_a_e[k, FracMomFields.m12]
                            - s_a_e[k, FracMomFields.m1]
                        )
                        H3a = (
                            s_a_e[k, FracMomFields.m12]
                            * s_a_e[k, FracMomFields.m32]
                            - s_a_e[k, FracMomFields.m1] ** 2
                        )
                        H3b = (
                            s_a_e[k, FracMomFields.m0]
                            - s_a_e[k, FracMomFields.m12]
                        ) * (
                            s_a_e[k, FracMomFields.m1]
                            - s_a_e[k, FracMomFields.m32]
                        ) - (
                            s_a_e[k, FracMomFields.m12]
                            - s_a_e[k, FracMomFields.m1]
                        ) ** 2

                        if s_a_e[k, FracMomFields.m0] < m0min:
                            theta_m0 = (
                                m0min - ucell[i, j, FracMomFields.m0]
                            ) / (
                                s_a_e[k, FracMomFields.m0]
                                - ucell[i, j, FracMomFields.m0]
                            )

                        if H1a < 0.0:
                            theta_m12a = (-ucell[i, j, FracMomFields.m12]) / (
                                s_a_e[k, FracMomFields.m12]
                                - ucell[i, j, FracMomFields.m12]
                            )

                        if H1b < 0.0:
                            theta_m12b = (
                                -ucell[i, j, FracMomFields.m0]
                                + ucell[i, j, FracMomFields.m12]
                            ) / (
                                s_a_e[k, FracMomFields.m0]
                                - ucell[i, j, FracMomFields.m0]
                                - s_a_e[k, FracMomFields.m12]
                                + ucell[i, j, FracMomFields.m12]
                            )

                        if H2a < 0.0:
                            a = (
                                s_a_e[k, FracMomFields.m0]
                                - ucell[i, j, FracMomFields.m0]
                            ) * (
                                s_a_e[k, FracMomFields.m1]
                                - ucell[i, j, FracMomFields.m1]
                            ) - (
                                s_a_e[k, FracMomFields.m12]
                                - ucell[i, j, FracMomFields.m12]
                            ) ** 2
                            b = (
                                ucell[i, j, FracMomFields.m0]
                                * (
                                    s_a_e[k, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m1]
                                )
                                + ucell[i, j, FracMomFields.m1]
                                * (
                                    s_a_e[k, FracMomFields.m0]
                                    - ucell[i, j, FracMomFields.m0]
                                )
                                - 2
                                * ucell[i, j, FracMomFields.m12]
                                * (
                                    s_a_e[k, FracMomFields.m12]
                                    - ucell[i, j, FracMomFields.m12]
                                )
                            )
                            c = (
                                ucell[i, j, FracMomFields.m0]
                                * ucell[i, j, FracMomFields.m1]
                                - ucell[i, j, FracMomFields.m12] ** 2
                            )
                            delta = b**2 - 4 * a * c
                            theta_m1a = max(
                                (-b + math.sqrt(delta)) / (2 * a),
                                (-b - math.sqrt(delta)) / (2 * a),
                            )

                        if H2b < 0.0:
                            theta_m1b = (
                                -ucell[i, j, FracMomFields.m12]
                                + ucell[i, j, FracMomFields.m1]
                            ) / (
                                s_a_e[k, FracMomFields.m12]
                                - ucell[i, j, FracMomFields.m12]
                                - s_a_e[k, FracMomFields.m1]
                                + ucell[i, j, FracMomFields.m1]
                            )

                        if H3a < 0.0:
                            a = (
                                s_a_e[k, FracMomFields.m12]
                                - ucell[i, j, FracMomFields.m12]
                            ) * (
                                s_a_e[k, FracMomFields.m32]
                                - ucell[i, j, FracMomFields.m32]
                            ) - (
                                s_a_e[k, FracMomFields.m1]
                                - ucell[i, j, FracMomFields.m1]
                            ) ** 2
                            b = (
                                ucell[i, j, FracMomFields.m12]
                                * (
                                    s_a_e[k, FracMomFields.m32]
                                    - ucell[i, j, FracMomFields.m32]
                                )
                                + ucell[i, j, FracMomFields.m32]
                                * (
                                    s_a_e[k, FracMomFields.m12]
                                    - ucell[i, j, FracMomFields.m12]
                                )
                                - 2
                                * ucell[i, j, FracMomFields.m1]
                                * (
                                    s_a_e[k, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m1]
                                )
                            )
                            c = (
                                ucell[i, j, FracMomFields.m12]
                                * ucell[i, j, FracMomFields.m32]
                                - ucell[i, j, FracMomFields.m1] ** 2
                            )
                            delta = b**2 - 4 * a * c
                            theta_m32a = max(
                                (-b + math.sqrt(delta)) / (2 * a),
                                (-b - math.sqrt(delta)) / (2 * a),
                            )

                        if H3b < 0.0:
                            a = (
                                s_a_e[k, FracMomFields.m0]
                                - ucell[i, j, FracMomFields.m0]
                                - s_a_e[k, FracMomFields.m12]
                                + ucell[i, j, FracMomFields.m12]
                            ) * (
                                s_a_e[k, FracMomFields.m1]
                                - ucell[i, j, FracMomFields.m1]
                                - s_a_e[k, FracMomFields.m32]
                                + ucell[i, j, FracMomFields.m32]
                            ) - (
                                s_a_e[k, FracMomFields.m12]
                                - ucell[i, j, FracMomFields.m12]
                                - s_a_e[k, FracMomFields.m1]
                                + ucell[i, j, FracMomFields.m1]
                            ) ** 2
                            b = (
                                (
                                    ucell[i, j, FracMomFields.m0]
                                    - ucell[i, j, FracMomFields.m12]
                                )
                                * (
                                    s_a_e[k, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m1]
                                    - s_a_e[k, FracMomFields.m32]
                                    + ucell[i, j, FracMomFields.m32]
                                )
                                + (
                                    ucell[i, j, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m32]
                                )
                                * (
                                    s_a_e[k, FracMomFields.m0]
                                    - ucell[i, j, FracMomFields.m0]
                                    - s_a_e[k, FracMomFields.m12]
                                    + ucell[i, j, FracMomFields.m12]
                                )
                                - 2
                                * (
                                    ucell[i, j, FracMomFields.m12]
                                    - ucell[i, j, FracMomFields.m1]
                                )
                                * (
                                    s_a_e[k, FracMomFields.m12]
                                    - ucell[i, j, FracMomFields.m12]
                                    - s_a_e[k, FracMomFields.m1]
                                    + ucell[i, j, FracMomFields.m1]
                                )
                            )
                            c = (
                                ucell[i, j, FracMomFields.m0]
                                - ucell[i, j, FracMomFields.m12]
                            ) * (
                                ucell[i, j, FracMomFields.m1]
                                - ucell[i, j, FracMomFields.m32]
                            ) - (
                                ucell[i, j, FracMomFields.m12]
                                - ucell[i, j, FracMomFields.m1]
                            ) ** 2
                            delta = b**2 - 4 * a * c

                            theta_m32b = max(
                                (-b + math.sqrt(delta)) / (2 * a),
                                (-b - math.sqrt(delta)) / (2 * a),
                            )

                        if (
                            s_a_e[k, FracMomFields.m1] * umax
                            - s_a_e[k, FracMomFields.m1U]
                            < 0.0
                        ):
                            theta_umax = (
                                -ucell[i, j, FracMomFields.m1] * umax
                                + ucell[i, j, FracMomFields.m1U]
                            ) / (
                                (
                                    s_a_e[k, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m1]
                                )
                                * umax
                                - s_a_e[k, FracMomFields.m1U]
                                + ucell[i, j, FracMomFields.m1U]
                            )

                        if (
                            s_a_e[k, FracMomFields.m1U]
                            - s_a_e[k, FracMomFields.m1] * umin
                            < 0.0
                        ):
                            theta_umin = (
                                -ucell[i, j, FracMomFields.m1U]
                                + ucell[i, j, FracMomFields.m1] * umin
                            ) / (
                                s_a_e[k, FracMomFields.m1U]
                                - ucell[i, j, FracMomFields.m1U]
                                - (
                                    s_a_e[k, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m1]
                                )
                                * umin
                            )
                        if (
                            s_a_e[k, FracMomFields.m1] * vmax
                            - s_a_e[k, FracMomFields.m1V]
                            < 0.0
                        ):
                            theta_vmax = (
                                -ucell[i, j, FracMomFields.m1] * vmax
                                + ucell[i, j, FracMomFields.m1V]
                            ) / (
                                (
                                    s_a_e[k, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m1]
                                )
                                * vmax
                                - s_a_e[k, FracMomFields.m1V]
                                + ucell[i, j, FracMomFields.m1V]
                            )

                        if (
                            s_a_e[k, FracMomFields.m1V]
                            - s_a_e[k, FracMomFields.m1] * vmin
                            < 0.0
                        ):
                            theta_vmin = (
                                -ucell[i, j, FracMomFields.m1V]
                                + ucell[i, j, FracMomFields.m1] * vmin
                            ) / (
                                s_a_e[k, FracMomFields.m1V]
                                - ucell[i, j, FracMomFields.m1V]
                                - (
                                    s_a_e[k, FracMomFields.m1]
                                    - ucell[i, j, FracMomFields.m1]
                                )
                                * vmin
                            )

                    theta[k] = max(
                        0.0,
                        min(
                            theta[k],
                            theta_m0,
                            theta_m12a,
                            theta_m12b,
                            theta_m1a,
                            theta_m1b,
                            theta_m32a,
                            theta_m32b,
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

        m1 = values[..., fields.m1]
        m1U = values[..., fields.m1U]
        m1V = values[..., fields.m1V]

        U = np.divide(m1U, m1)
        V = np.divide(m1V, m1)

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
        maxvel = np.max(np.abs(U))

        # Min mesh dx
        dx = cells.min_length

        new_dt = CFL_value * dx / maxvel

        return np.min((dt, new_dt))
