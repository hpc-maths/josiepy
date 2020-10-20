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
import abc
import numpy as np

from josie.mesh.cellset import CellSet, MeshCellSet

from .scheme import Scheme


class NonConservativeScheme(Scheme):
    r"""A mixin that provides the scheme implementation for the non
    conservative term
    """

    def accumulate(self, cells: MeshCellSet, neighs: CellSet, t: float):

        # Accumulate other terms
        super().accumulate(cells, neighs, t)

        B = self.problem.B(cells)
        G = self.G(cells, neighs)
        BG = np.einsum("...ijk,...k->...i", B, G)

        self._fluxes += BG

    @abc.abstractmethod
    def G(self, cells: MeshCellSet, neighs: CellSet) -> np.ndarray:

        r"""This is the non-conservative flux implementation of the scheme. See
        :cite:`toro_riemann_2009` for a great overview on numerical methods for
        hyperbolic problems.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull

        The non-conservative term is discretized as follows:

        .. math::

            \numNonConservativeFull

        A concrete instance of this class needs to implement the discretization
        of the numerical flux on **one** face of a cell. It needs to implement
        the term :math:`\numNonConservative`


        Parameters
        ----------
        values
            The values of the state fields in each cell

        neighs
            A :class:`CellSet` containing data of neighbour cells corresponding
            to the :attr:`values`

        Returns
        -------
        G
            The value of the numerical nonconservative flux multiplied by
            the surface value :math:`\numNonConservative`
        """

        pass
