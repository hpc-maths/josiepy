# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .schemes import TsCapScheme
from .state import Q, TsCapConsFields, TsCapConsState
from josie.twofluid.fields import Phases
from ..dimension import MAX_DIMENSIONALITY


class ArithmeticCap(TsCapScheme):
    def post_extrapolation(self, values: Q):
        # self.prim2Q(values)
        # auxilliary variables update
        self.auxilliaryVariableUpdateNoGeo(values)

    def intercellFlux(
        self,
        Q_L: Q,
        Q_R: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        r"""Exact solver scheme

        Parameters
        ----------
        cells:
            A :class:`MeshCellSet` containing the state of the mesh cells

        neighs
            A :class:`NeighboursCellSet` containing data of neighbour cells
            corresponding to the :attr:`values`

        Returns
        -------
        F
            The value of the numerical convective flux multiplied by the
            surface value :math:`\numConvective`
        """
        FS = np.zeros_like(Q_L).view(Q)
        # Compute arithmetic mean of gemetric variables norm_grad_a, n_x, n_y
        intercells = Q_L
        intercells[..., Q.fields.norm_grada] = np.where(
            (Q_L[..., Q.fields.norm_grada] > 0) & (Q_R[..., Q.fields.norm_grada] > 0),
            0.5 * (Q_L[..., Q.fields.norm_grada] + Q_R[..., Q.fields.norm_grada]),
            0,
        )
        intercells[..., Q.fields.n_x] = np.where(
            intercells[..., Q.fields.norm_grada] > 0,
            0.5 * (Q_L[..., Q.fields.n_x] + Q_R[..., Q.fields.n_x]),
            0,
        )
        intercells[..., Q.fields.n_y] = np.where(
            intercells[..., Q.fields.norm_grada] > 0,
            0.5 * (Q_L[..., Q.fields.n_y] + Q_R[..., Q.fields.n_y]),
            0,
        )
        F = np.einsum("...mkl,...l->...mk", self.problem.F_cap(intercells), normals)

        # Multiply by surfaces
        FS.set_conservative(surfaces[..., np.newaxis, np.newaxis] * F)

        return FS
