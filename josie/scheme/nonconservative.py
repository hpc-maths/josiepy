# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import numpy as np

from ..mesh.cellset import NeighboursCellSet, MeshCellSet

from .scheme import Scheme
from ..problem import NonConservativeProblem


class NonConservativeScheme(Scheme):
    r"""A mixin that provides the scheme implementation for the non
    conservative term
    """

    problem: NonConservativeProblem

    def accumulate(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float):
        # Accumulate other terms
        super().accumulate(cells, neighs, t)

        B = self.problem.B(cells.values)
        G = self.G(cells, neighs)

        BG = np.einsum("...ikl,...kl->...i", B, G)

        # FIXME: Ignoring typing: https://github.com/numpy/numpy/issues/20072
        self._fluxes += BG  # type: ignore

    @abc.abstractmethod
    def G(self, cells: MeshCellSet, neighs: NeighboursCellSet) -> np.ndarray:
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
            A :class:`NeighboursCellSet` containing data of neighbour cells
            corresponding to the :attr:`values`

        Returns
        -------
        G
            The value of the numerical nonconservative flux multiplied by
            the surface value :math:`\numNonConservative`
        """

        pass
