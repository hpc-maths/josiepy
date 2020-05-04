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
import copy
import numpy as np

from typing import Callable, Tuple, TYPE_CHECKING, Union

from josie.solver.solver import Ghost, State
from .geom import BoundaryCurve
from .math import Direction

if TYPE_CHECKING:

    # This is a trick to enable mypy to evaluate the Enum as a standard
    # library Enum for type checking but we use `aenum` in the running code
    from enum import Enum  # pragma: no cover

    NoAlias = object()  # pragma: no cover
else:
    from aenum import Enum, NoAlias


class BoundaryCondition:
    """ A :class:`BoundaryCondition` is a collection of
    :class:`ScalarBC`, one per each :attr:`~.State.fields` of
    the :class:`~.Problem`.

    Attributes
    ----------
    bc
        A :class:`State` instance whose elements per each field are not `float`
        but :class:`ScalarBC` instead

        >>> from josie.solver.state import StateTemplate
        >>> MyState = StateTemplate("u", "v")
        >>> mybc = BoundaryCondition(MyState(Dirichlet(0), Neumann(1)))
    """

    def __init__(self, bc: State):
        self.bc = bc

    def __call__(self, ghost: Ghost, t: float = 0):
        """
        Parameters
        ----------
        boundary_idx
            Indices of the cells of the boundary to which the
            :class:`BoundaryCondition` is applied to
        ghost_idx
            Indices of the ghost cells associated to the cells indexed by
            ``boundary_idx``
        solver
            The solver object that can be useful to use other cells of the mesh
            (probably needed only for :class:`Periodic`) and to access the
            :class:`Mesh` and the field values
        t
            The time instant to which this :class:`ScalarBC`
            must be evaluated (useful for time-dependent BCs)

        """
        # Apply BC for each field
        for field in self.bc.fields:
            centroids = ghost.get_boundary_centroids()
            boundary_values = ghost.get_boundary_values(field)

            ghost.set_values(
                field, self.bc[field](centroids, boundary_values, t)
            )


class ScalarBC(abc.ABC):
    """ A :class:`ScalarBC` is implemented as a callable that
    returns an equivalent cell value for the specific field for each cell given
    to it.

    This returned values can be an actual value that ensure the value of the
    :class:`State` or of its gradient, or they can just be "pointers" to
    internal cells as in the case of of Periodic, that returns the
    corresponding cell on the opposite boundary.
    """

    @abc.abstractmethod
    def __call__(
        self, centroids: np.ndarray, values: np.ndarray, t: float = 0,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        centroids
            An array containing the cell centroids of the cells on the boundary
            which this :class:`ScalarBC`is applied to
        values
            An array containing the field values of the cells
            on the boundary which this ScalarBC is applied to
        t
            The time instant to which this :class:`ScalarBC`
            must be evaluated (useful for time-dependent BCs)

        Returns
        -------
        The value of the field on the ghost cell

        """
        raise NotImplementedError


class Dirichlet(ScalarBC):
    r""" A :class:`ScalarBC` that fixes a value of a field on the boundary.

    Assuming we want to impose the value :math:`Q = Q_D` on the (left, as an
    example) boundary, we can assume that the value on the boundary is
    approximated by:

    .. math::

        \frac{Q_{0,j} + Q_\text{ghost}}{2} = Q_D

    That means we can impose the :class:`ScalarBC` assigning the
    value of

    .. math::

        Q_\text{ghost} = 2Q_D - Q_{0,j}

    to the ghost cell.

    Parameters
    ----------
    value
        The value of the field to be imposed on the boundary
    """
    _value: float

    def __new__(cls, value: Union[State, float]):  # type: ignore
        if isinstance(value, State):
            # An entire state was provided as value of the Dirichlet BC, so
            # we return a BoundaryCondition with a Dirichlet BC for each
            # field with the right value
            bcs = []

            for field in value.fields:
                bcs.append(cls.__new__(cls, value[field]))

            return BoundaryCondition(np.array(bcs).view(type(value)))
        else:
            obj = super().__new__(cls)
            obj._value = value
            return obj

    def __call__(
        self, centroids: np.ndarray, values: np.ndarray, t: float = 0,
    ) -> np.ndarray:

        return 2 * self._value - values


class Neumann(Dirichlet):
    r""" A :class:`ScalarBC` that fixes a value of the gradient of a field on
    the boundary.

    Assuming we want to impose the value of the gradient :math:`\frac{\partial
    Q}{\partial \hat{\mathbf{n}}} = Q_N` of the state on the (left, as an
    example) boundary, we can assume that the value of the gradient on the
    boundary is approximated by:

    .. math::

        \frac{Q_{0,j} - Q_\text{ghost}}{\Delta x} = Q_N

    That means we can impose the :class:`ScalarBC` assigning the
    value of

    .. math::

        Q_\text{ghost} = Q_{0,j} - Q_N

    to the ghost cell (assuming a :math:`\Delta x = 1`)

    Parameters
    ----------
    value
        The value of the gradient of the field to be imposed on the boundary
    """

    def __call__(
        self, centroids: np.ndarray, values: np.ndarray, t: float = 0,
    ) -> np.ndarray:
        return values - self._value


class NeumannDirichlet(ScalarBC):
    """ A :class:`ScalarBC` that applies a :class:`Dirichlet` on a part of the
    domain and a :class:`Neumann` on the rest of it.

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

    neumann_value: float
    dirichlet_value: float
    partition_fun: Callable[[NeumannDirichlet, np.ndarray], np.ndarray]

    def __new__(
        cls,
        neumann_value: Union[State, float],
        dirichlet_value: Union[State, float],
        partition_fun: Callable[[np.ndarray], np.ndarray],
    ):
        if isinstance(neumann_value, State) and isinstance(
            dirichlet_value, State
        ):
            # An entire state was provided as value of the Dirichlet BC, so
            # we return a BoundaryCondition with a Dirichlet BC for each
            # field with the right value
            bcs = []

            for field in neumann_value.fields:
                bcs.append(
                    cls.__new__(
                        cls,
                        neumann_value[field],
                        dirichlet_value[field],
                        partition_fun,
                    )
                )

            return BoundaryCondition(np.array(bcs).view(type(neumann_value)))
        else:
            obj = super().__new__(cls)
            obj.neumann_value = neumann_value
            obj.dirichlet_value = dirichlet_value
            obj.partition_fun = partition_fun
            return obj

    def __call__(
        self, centroids: np.ndarray, values: np.ndarray, t: float = 0,
    ) -> np.ndarray:
        # First apply Neumann to everything
        ghost_values = values - self.neumann_value

        # Then extract the cells where to apply the Dirichlet BC
        # FIXME: This type check ignore should probably be fixed by mypy
        dirichlet_cells = self.partition_fun(centroids)

        ghost_values[dirichlet_cells] = (
            2 * self.dirichlet_value - values[dirichlet_cells]
        )

        return ghost_values


class Side(Enum, settings=NoAlias):
    """ A Enum encapsulating the 4 possibilities of a :class:`Periodic`
    :class:`ScalarBC` """

    LEFT = -1
    BOTTOM = -1
    RIGHT = 0
    TOP = 0


class Periodic(BoundaryCondition):
    r""" A :class:`BoundaryCondition` that connects one side of the domain to
    the other. In general is more straighforward to use the
    :func`make_periodic` function on a couple of :class:`~.BoundaryCurve` that
    needs to be linked periodically

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

    Raises
    ------
    ValueError
        If `side` is not recognized
    """

    def __init__(self, side: Side):
        self._side = side

    def __call__(self, ghost: Ghost, t: float = 0):

        if self._side in [Side.LEFT, Side.RIGHT]:
            periodic_idx = (self._side.value, slice(None))
        elif self._side in [Side.BOTTOM, Side.TOP]:
            periodic_idx = (slice(None), self._side.value)
        else:
            raise ValueError(f"Unknown side. Expecting a {Side} object")

        ghost.solver._values[ghost.ghost_cells_idx] = copy.deepcopy(
            ghost.solver.values[periodic_idx]
        )


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
