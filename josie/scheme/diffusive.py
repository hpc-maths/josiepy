# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from ..scheme import Scheme
from ..mesh.cellset import MeshCellSet, NeighboursCellSet
from ..problem import DiffusiveProblem


class DiffusiveScheme(Scheme):
    """A mixin that provides the scheme interface for the diffusive term. The
    :class:`DiffusiveScheme` needs to implement a strategy to approximate the
    state gradient at the cell interface with its neighbour"""

    problem: DiffusiveProblem

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

    def CFL(self, cells: MeshCellSet, CFL_value: float) -> float:
        r"""Definition of CFL for a parabolic problem

        .. math::

            \dd{t} = C_{fl} \frac{\dd{x}^2}{2 \mu}

        """
        dt = super().CFL(cells, CFL_value)

        # Min mesh dx
        dx = cells.min_length

        viscosity = np.max(self.problem.K(cells))

        new_dt = CFL_value * (dx**2) / 2 / viscosity

        return np.min((dt, new_dt))

    def accumulate(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float):
        # Compute fluxes computed eventually by the other terms
        super().accumulate(cells, neighs, t)

        # Add conservative contribution
        # FIXME: Ignoring typing: https://github.com/numpy/numpy/issues/20072
        self._fluxes -= self.D(cells, neighs)  # type: ignore
