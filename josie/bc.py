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

import abc


from typing import Tuple, TYPE_CHECKING

from .mesh.cell import Cell, GhostCell
from .mesh.mesh import Mesh
from .geom import BoundaryCurve

if TYPE_CHECKING:
    from josie.solver.state import State

    # This is a trick to enable mypy to evaluate the Enum as a standard
    # library Enum for type checking but we use `aenum` in the running code
    from enum import Enum, auto
    NoAlias = object()
else:
    from aenum import Enum, NoAlias, auto


class BoundaryCondition(metaclass=abc.ABCMeta):
    """ A BoundaryCondition is implemented as a callable that returns an
    equivalent cell that is associated as a NeighbourCell to the cells on the
    boundary.

    This returned cell can be an actual `Cell` as in the case
    of Periodic, that returns the corresponding cell on the opposite boundary,
    or a GhostCell, that means a cell that only stores a value that is then
    used by the numerical scheme.
    """

    @abc.abstractmethod
    def __call__(self, mesh: Mesh, cell: Cell) -> 'GhostCell':
        raise NotImplementedError


class Dirichlet(BoundaryCondition):
    r""" A Dirichlet BoundaryCondition is a BoundaryCondition that fixes a
    value of the State on the boundary.

    Assuming we want to impose the value :math:`Q = Q_D` on the (left, as an
    example) boundary, we can assume that the value on the boundary is
    approximated by:

    .. math::

    \frac{Q_{0,j} + Q_\text{ghost}}{2} = Q_D

    That means we can impose the BoundaryCondition assigning the value of

    .. math::
    Q_\text{ghost} = 2Q_D - \frac{Q_{0,j}}

    to the ghost cell.

    Parameters
    ----------
    value
        The value of the state to be imposed on the boundary
    """

    def __init__(self, value: 'State'):
        self._value = value

    def __call__(self, mesh: Mesh, cell: Cell) -> 'GhostCell':
        return GhostCell(2*self._value - cell.value)  # type: ignore


class Neumann(Dirichlet):
    r""" A Neumann BoundaryCondition is a BoundaryCondition that fixes a
    value of the gradient of the State on the boundary.

    Assuming we want to impose the value of the gradient :math:`\frac{\partial
    Q}{\partial \hat{\mathbf{n}}} = Q_N`of the state on the (left, as an
    example) boundary, we can assume that the value of the gradient on the
    boundary is approximated by:

    .. math::

    \frac{Q_{0,j} - Q_\text{ghost}}{\Delta x} = Q_N

    That means we can impose the BoundaryCondition assigning the value of

    .. math::
    Q_\text{ghost} = \frac{Q_{0,j}} - Q_N

    to the ghost cell (assuming a :math:`\Delta x = 1`)

    Parameters
    ----------
    value
        The value of the state to be imposed on the boundary
    """

    def __call__(self, mesh: Mesh, cell: Cell) -> 'GhostCell':
        return GhostCell(cell.value - self._value)  # type: ignore


class Side(Enum, settings=NoAlias):
    """ A Enum encapsulating the 4 possibilities of a Periodic
    BoundaryCondition """

    LEFT = -1
    BOTTOM = -1
    RIGHT = 0
    TOP = 0


class Direction(Enum):
    """ An Enum encapsulating the direction of a Periodic BoundaryCondition
    """
    X = auto()
    Y = auto()


class Periodic(BoundaryCondition):
    r""" A Periodic BoundaryCondition is a BoundaryCondition that connects
    one side of the domain to the other. In general is more straighforward
    to use the `make_periodic` function on a couple of BoundaryCurve that
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
    """

    def __init__(self, side: Side):
        self._side = side

    def __call__(self, mesh: Mesh, cell: Cell) -> 'Cell':

        if self._side in [Side.LEFT, Side.RIGHT]:
            return mesh.cells[self._side.value, cell.j]
        elif self._side in [Side.BOTTOM, Side.TOP]:
            return mesh.cells[cell.i, self._side.value]
        else:
            raise ValueError(f'Unknown side. Expecting a {Side} object')


def make_periodic(first: BoundaryCurve, second: BoundaryCurve,
                  direction: Direction) \
        -> Tuple[BoundaryCurve, BoundaryCurve]:
    """ This handy function takes as arguments two opposed BoundaryCurve and
    configures them correctly to provide periodic behaviour.

    Parameters
    ----------
    first
        The first BoundaryCurve to link
    second
        The second BoundaryCurve to link
    direction
        The direction on which the two BoundaryCurve are periodically
        connected

    Returns
    -------
    first
        The first BoundaryCurve whose `bc` attribute is correctly configured
    second
        The second BoundaryCurve whose `bc` attribute is correctly configured
    """

    if direction is Direction.X:
        first.bc = Periodic(Side.LEFT)
        second.bc = Periodic(Side.RIGHT)
    elif direction is Direction.Y:
        first.bc = Periodic(Side.BOTTOM)
        second.bc = Periodic(Side.TOP)
    else:
        raise ValueError(f'Unknown direction. Expecting a {Direction} object')

    return first, second
