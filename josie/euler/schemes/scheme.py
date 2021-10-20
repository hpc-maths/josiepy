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

from typing import TYPE_CHECKING

from josie.euler.problem import EulerProblem
from josie.euler.state import EulerState
from josie.scheme.convective import ConvectiveScheme


if TYPE_CHECKING:
    from josie.euler.eos import EOS
    from josie.mesh.cellset import MeshCellSet
    from josie.fluid.state import SingleFluidState


class EulerScheme(ConvectiveScheme):
    """A general base class for Euler schemes"""

    problem: EulerProblem

    def __init__(self, eos: EOS):
        super().__init__(EulerProblem(eos))

    def post_step(self, cells: MeshCellSet):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        values: EulerState = cells.values.view(EulerState)

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
        values[..., fields.e] = e

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
