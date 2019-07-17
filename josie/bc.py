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

from enum import Enum, auto

from typing import Tuple

from .mesh.cell import Cell
from .mesh.mesh import Mesh
from .geom import BoundaryCurve

from josie.solver.state import State


class BoundaryCondition(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, mesh: Mesh, cell: Cell) -> State:
        raise NotImplementedError


class Dirichlet(BoundaryCondition):
    def __init__(self, value: State):
        self._value = value

    def __call__(self, mesh: Mesh, cell: Cell) -> State:
        return 2*self._value - cell.new


class Neumann(Dirichlet):

    def __call__(self, mesh: Mesh, cell: Cell) -> State:
        return self._value + cell.new


class Side(Enum):
    LEFT = -1
    BOTTOM = -1
    RIGHT = 0
    TOP = 0


class Direction(Enum):
    X = auto()
    Y = auto()


class Periodic(BoundaryCondition):

    def __init__(self, side: Side):
        self._side = side

    def __call__(self, mesh: Mesh, cell: Cell) -> float:

        if self._side in [Side.LEFT, Side.RIGHT]:
            return mesh.cells[self._side.value, cell.j].value
        elif self._side in [Side.BOTTOM, Side.TOP]:
            return mesh.cells[cell.i, self._side.value].value
        else:
            raise ValueError(f'Unknown side. Expecting a {Side} object')


def make_periodic(first: BoundaryCurve, second: BoundaryCurve,
                  direction: Direction) \
        -> Tuple[BoundaryCurve]:

    if direction is Direction.X:
        first.bc = Periodic(Side.LEFT)
        second.bc = Periodic(Side.RIGHT)
    elif direction is Direction.Y:
        first.bc = Periodic(Side.BOTTOM)
        second.bc = Periodic(Side.TOP)
    else:
        raise ValueError(f'Unknown direction. Expecting a {Direction} object')

    return first, second
