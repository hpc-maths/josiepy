# josiepy
# Copyright © 2021 Ruben Di Battista
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
import numpy as np

from josie.scheme import Scheme
from josie.mesh.cellset import MeshCellSet, NeighboursCellSet


class DiffusiveScheme(Scheme):
    """A mixin that provides the scheme interface for the diffusive term. The
    :class:`DiffusiveScheme` needs to implement a strategy to approximate the
    state gradient at the cell interface with its neighbour"""

    def D(self, cells: MeshCellSet, neighs: NeighboursCellSet) -> np.ndarray:
        r"""This is the diffusive flux implementation of the scheme. See
        :cite:`toro_riemann_2009` for a great overview on numerical methods for
        hyperbolic problems.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull

        The diffusive term is discretized as follows:

        .. math::

            \numDiffusiveFull


        A concrete implementation of this method needs to implement the
        discretization of the numerical diffusive flux on **one** face of a
        cell. It needs to implement the term :math:`\numDiffusive`

        Parameters
        ----------
        values
            The values of the state fields in each cell

        neighs
            A :class:`NeighboursCellSet` containing data of neighbour cells
            corresponding to the :attr:`values`

        Returns
        -------
        D
            The value of the numerical diffusive flux multiplied by
            the surface value :math:`\numDiffusive`
        """

        raise NotImplementedError

    def accumulate(
        self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float
    ):

        # Compute fluxes computed eventually by the other terms
        super().accumulate(cells, neighs, t)

        # Add conservative contribution
        self._fluxes -= self.D(cells, neighs)
        #
