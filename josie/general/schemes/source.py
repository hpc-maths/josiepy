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

    def pre_accumulate(self, cells: MeshCellSet, t: float):
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

        super().pre_accumulate(cells, t)

        self._fluxes += self.volume_s(cells, t)

    def accumulate(
        self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float
    ) -> State:
        """We do not use the :meth:`accumulate` method to put source
        contribution into the fluxes but we do in in :meth:`pre_step`, because
        we do not need to do it for each face of the cell"""

        return np.zeros_like(cells.values)

    def volume_s(self, cells: MeshCellSet, t: float) -> State:
        """The source flux computed only for the cells, without taking into
        consideration the neighbours since they're not needed
        """
        return self.problem.s(cells, t) * cells.volumes[..., np.newaxis]

    def s(
        self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float
    ) -> State:
        """ Use :meth:`volume_s` instead """
        raise NotImplementedError
