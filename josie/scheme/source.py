# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc

from typing import TYPE_CHECKING

from .scheme import Scheme
from ..problem import SourceProblem

if TYPE_CHECKING:
    from josie.mesh.cellset import NeighboursCellSet, MeshCellSet
    from josie.state import State


class SourceScheme(Scheme):
    r"""A mixin that provides the scheme implementation for the source term

    .. math::

        \numSourceFull
    """

    problem: SourceProblem

    def accumulate(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float):
        # Compute fluxes computed eventually by the other terms (diffusive,
        # nonconservative, source)
        super().accumulate(cells, neighs, t)

        # Add conservative contribution
        # FIXME: Ignoring typing: https://github.com/numpy/numpy/issues/20072
        self._fluxes += self.s(cells, neighs, t)  # type: ignore

    @abc.abstractmethod
    def s(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float) -> State:
        r"""This is the source term implementation of the scheme. See
        :cite:`toro_riemann_2009` for a great overview on numerical methods for
        hyperbolic problems.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull

        The source term is discretized as follows:

        .. math::

            \numSourceFull


        Parameters
        ----------
        cells:
            A :class:`MeshCellSet` containing the state of the mesh cells

        neighs
            A :class:`NeighboursCellSet` containing data of neighbour cells
            corresponding to the :attr:`values`

        t
            Time instant

        Returns
        -------
        s
            The value of the source term approximated in the cell
        """

        pass
