# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from josie.scheme.source import SourceScheme

if TYPE_CHECKING:
    from josie.mesh.cellset import NeighboursCellSet, MeshCellSet
    from josie.state import State


class ConstantSource(SourceScheme):
    r"""A mixing that provides the approximation of the source term as a constant
    value computed at the centroid of the cell.

    .. math::

        \numSource \sim \abs{\pdeSource}_i V_i

    """

    def pre_accumulate(self, cells: MeshCellSet, dt: float, t: float):
        """
        We add the source term flux here since we just need cell info and not
        neighbours info

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells

        t
            The current time instant
        """

        super().pre_accumulate(cells, dt, t)

        # FIXME: Ignoring typing: https://github.com/numpy/numpy/issues/20072
        self._fluxes += self.volume_s(cells, t)  # type: ignore

    def accumulate(
        self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float
    ) -> State:
        """We do not use the :meth:`accumulate` method to put source
        contribution into the fluxes but we do in in :meth:`pre_accumulate`,
        because we do not need to do it for each face of the cell"""

        return np.zeros_like(cells.values)

    def volume_s(self, cells: MeshCellSet, t: float) -> State:
        """The source flux computed only for the cells, without taking into
        consideration the neighbours since they're not needed
        """
        return self.problem.s(cells, t) * cells.volumes[..., np.newaxis, np.newaxis]

    def s(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float) -> State:
        """Use :meth:`volume_s` instead"""
        raise NotImplementedError
