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

from .mesh.cell import Cell
from .mesh.mesh import Mesh
from .geom import BoundaryCurve


class BoundaryCondition(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, mesh: Mesh, cell: Cell) -> float:
        raise NotImplementedError


class Dirichlet(BoundaryCondition):
    def __init__(self, value):
        self._value = value

    def __call__(self, mesh: Mesh, cell: Cell) -> float:
        return 2*self._value - cell.new


class Neumann(Dirichlet):

    def __call__(self, mesh: Mesh, cell: Cell) -> float:
        return self._value + cell.new


class Periodic(BoundaryCondition):

    # First number indicates i: 0 or j:1. Second value of the tuple indicates
    # the actual index value.
    # TODO: probably better implemented with an enum
    corresponding_cell_idx = {
        'left': (0, -1),
        'bottom': (1, -1),
        'right': (0, 0),
        'top': (1, 0),
    }

    def __init__(self, side: 'enum'):
        self._side = side

    def __call__(self, mesh: Mesh, cell: Cell) -> float:
        i, ival = corresponding_cell_idx[self._side]

        if i == 0:
            return mesh.cells[ival, cell.j].new
        else:
            return mesh.cells[cell.i, ival].new
