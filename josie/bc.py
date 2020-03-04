# josiepy
# Copyright © 2019 Ruben Di Battista
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

import abc
import numpy as np

from typing import Callable, Tuple, TYPE_CHECKING

from josie.solver.solver import Solver
from .geom import BoundaryCurve
from .math import Direction

if TYPE_CHECKING:
    from josie.solver.state import State  # pragma: no cover

    # This is a trick to enable mypy to evaluate the Enum as a standard
    # library Enum for type checking but we use `aenum` in the running code
    from enum import Enum  # pragma: no cover

    NoAlias = object()  # pragma: no cover
else:
    from aenum import Enum, NoAlias


class BoundaryCondition(metaclass=abc.ABCMeta):
    """ A :class:`BoundaryCondition` is implemented as a callable that returns
    an equivalent cell value for each cell given to it.

    This returned values can be an actual values that ensure the value of the
    :class:`State` or of its gradient, or they can just be "pointers" to
    internal cells as in the case of of Periodic, that returns the
    corresponding cell on the opposite boundary.
    """

    @abc.abstractmethod
    def __call__(
        self,
        solver: Solver,
        centroids: np.ndarray,
        values: np.ndarray,
        t: float = 0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        mesh
            The mesh object that can be useful to use other cells of the mesh
            (probably needed only for :class:`Periodic`)
        centroids
            An array containing the cell centroids of the cells on the boundary
            which this BoundaryCondition is applied to
        values
            An array containing the cell :class:`~.State` values of the cells
            on the boundary which this BoundaryCondition is applied to
        t
            The time instant to which this BoundaryCondition must be evaluated
            (useful for time-dependent BCs)

        """
        raise NotImplementedError


class Dirichlet(BoundaryCondition):
    r""" A :class:`BoundaryCondition` that fixes a
    value of the State on the boundary.

    Assuming we want to impose the value :math:`Q = Q_D` on the (left, as an
    example) boundary, we can assume that the value on the boundary is
    approximated by:

    .. math::

        \frac{Q_{0,j} + Q_\text{ghost}}{2} = Q_D

    That means we can impose the :class:`BoundaryCondition` assigning the value
    of

    .. math::

        Q_\text{ghost} = 2Q_D - Q_{0,j}

    to the ghost cell.

    Parameters
    ----------
    value
        The value of the state to be imposed on the boundary
    """

    def __init__(self, value: State):
        self._value = value

    def __call__(
        self,
        solver: Solver,
        centroids: np.ndarray,
        values: np.ndarray,
        t: float = 0,
    ) -> np.ndarray:

        return 2 * self._value - values


class Neumann(Dirichlet):
    r""" A :class:`BoundaryCondition` that fixes a
    value of the gradient of the State on the boundary.

    Assuming we want to impose the value of the gradient :math:`\frac{\partial
    Q}{\partial \hat{\mathbf{n}}} = Q_N` of the state on the (left, as an
    example) boundary, we can assume that the value of the gradient on the
    boundary is approximated by:

    .. math::

        \frac{Q_{0,j} - Q_\text{ghost}}{\Delta x} = Q_N

    That means we can impose the :class:`BoundaryCondition` assigning the value
    of

    .. math::

        Q_\text{ghost} = Q_{0,j} - Q_N

    to the ghost cell (assuming a :math:`\Delta x = 1`)

    Parameters
    ----------
    value
        The value of the state to be imposed on the boundary
    """

    def __call__(
        self,
        solver: Solver,
        centroids: np.ndarray,
        values: np.ndarray,
        t: float = 0,
    ) -> np.ndarray:
        return values - self._value


class NeumannDirichlet(BoundaryCondition):
    """ A :class:`BoundaryCondition` that applies a :class:`Dirichlet`
    :class:`BoundaryCondition` on a part of the domain and a :class:`Neumann`
    on the rest of it.

    Parameters
    ---------
    neumann_value
        The value of the gradient to use for the :class:`Neumann` partition of
        the boundary
    dirichlet_value
        The value of the :class:`State` to use for the :class:`Dirichlet`
        partition of the boundary
    partition_fun
        A callable that takes as input the coordinates of the boundary cells
        and returns the indices of the cells for which the :class:`Dirichlet`
        boundary condition has to be applied
    """

    def __init__(
        self,
        neumann_value: State,
        dirichlet_value: State,
        partition_fun: Callable[[np.ndarray], np.ndarray],
    ):
        self.neumann_value = neumann_value
        self.dirichlet_value = dirichlet_value
        self.partition_fun = partition_fun

    def __call__(
        self,
        solver: Solver,
        centroids: np.ndarray,
        values: np.ndarray,
        t: float = 0,
    ) -> np.ndarray:
        # First apply Neumann to everything
        ghost_values = values - self.neumann_value

        # Then extract the cells where to apply the Dirichlet BC
        dirichlet_cells = self.partition_fun(centroids)

        ghost_values[dirichlet_cells] = (
            2 * self.dirichlet_value - values[dirichlet_cells]
        )

        return ghost_values


class Side(Enum, settings=NoAlias):
    """ A Enum encapsulating the 4 possibilities of a :class:`Periodic`
    :class:`BoundaryCondition` """

    LEFT = -1
    BOTTOM = -1
    RIGHT = 0
    TOP = 0


class Periodic(BoundaryCondition):
    r""" A Periodic BoundaryCondition is a :class:`~.BoundaryCondition` that
    connects one side of the domain to the other. In general is more
    straighforward to use the ``make_periodic`` function on a couple of
    :class:`~.BoundaryCurve` that needs to be linked periodically

    That means the neighbour cells of the cells on one domain are the cells
    on the other side of the domain. In other words, as an example, given a
    cell on the left boundary, identified by the indices :math:`i,j = (0, j)`,
    i.e. :math:`C_{0,j}`, its west neighbour cell needs to be :math:`C_{N, j}`,
    being :math:`N` the number of cells along the :math:`x`-direction (i.e.
    for increasing index :math:`i`)

    Parameters
    ----------
    side
        The side on which the Periodic BC is configured
    """

    def __init__(self, side: Side):
        self._side = side

    def __call__(
        self,
        solver: Solver,
        centroids: np.ndarray,
        values: np.ndarray,
        t: float = 0,
    ) -> np.ndarray:

        if self._side in [Side.LEFT, Side.RIGHT]:
            return solver.values[self._side.value, :]
        elif self._side in [Side.BOTTOM, Side.TOP]:
            return solver.values[:, self._side.value]
        else:
            raise ValueError(f"Unknown side. Expecting a {Side} object")


def make_periodic(
    first: BoundaryCurve, second: BoundaryCurve, direction: Direction
) -> Tuple[BoundaryCurve, BoundaryCurve]:
    """ This handy function takes as arguments two opposed BoundaryCurve and
    configures them correctly to provide periodic behaviour.

    Parameters
    ----------
    first
        The first :class:`~.BoundaryCurve` to link
    second
        The second :class:`~.BoundaryCurve` to link
    direction
        The direction on which the two :class:`~.BoundaryCurve` are
        periodically connected

    Returns
    -------
    first
        The first :class:`~.BoundaryCurve` whose `bc` attribute is correctly
        configured
    second
        The second :class:`~.BoundaryCurve` whose `bc` attribute is correctly
        configured
    """

    if direction is Direction.X:
        first.bc = Periodic(Side.LEFT)
        second.bc = Periodic(Side.RIGHT)
    elif direction is Direction.Y:
        first.bc = Periodic(Side.BOTTOM)
        second.bc = Periodic(Side.TOP)
    else:
        raise ValueError(f"Unknown direction. Expecting a {Direction} object")

    return first, second
