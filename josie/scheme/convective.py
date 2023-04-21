# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc

from typing import TYPE_CHECKING

from .scheme import Scheme
from ..problem import ConvectiveProblem

import numpy as np


if TYPE_CHECKING:
    from josie.mesh.cellset import NeighboursCellSet, MeshCellSet
    from josie.state import State


class ConvectiveScheme(Scheme):
    r"""A mixin that provides the scheme implementation for the convective
    term

    .. math::

        \numConvectiveFull

    """

    problem: ConvectiveProblem

    @abc.abstractmethod
    def intercellFlux(
        self,
        Q_L,
        Q_R,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet) -> State:
        r"""This is the convective flux implementation of the scheme. See
        :cite:`toro_riemann_2009` for a great overview on numerical methods for
        hyperbolic problems.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull

        The convective term is discretized as follows:

        .. math::

            \numConvectiveFull

        A concrete implementation of this method needs to implement the
        discretization of the numerical flux on **one** face of a cell. It
        needs to implement the term :math:`\numConvective`


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
        raise NotImplementedError

    def limiter(self, slope_L: np.ndarray, slope_R: np.ndarray):
        pass

    def accumulate(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float):
        # Compute fluxes computed eventually by the other terms (diffusive,
        # nonconservative, source)
        super().accumulate(cells, neighs, t)

        # Add conservative contribution
        # FIXME: Ignoring typing: https://github.com/numpy/numpy/issues/20072
        self._fluxes += self.F(cells, neighs)  # type: ignore
