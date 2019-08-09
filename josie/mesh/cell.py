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
import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from josie.solver.state import State  # noqa: F401


PointType = Union[Tuple[float, float], np.ndarray]


class GhostCell:
    def __init__(self, value: 'State'):
        self.value = value


class Cell(GhostCell):
    def __init__(self,
                 nw: PointType, sw: PointType,
                 se: PointType, ne: PointType,
                 i: int, j: int, value: 'State'):

        self.nw = np.asarray(nw)
        self.sw = np.asarray(sw)
        self.se = np.asarray(se)
        self.ne = np.asarray(ne)

        self.faces = [
            Face(self.nw, self.sw),
            Face(self.sw, self.se),
            Face(self.se, self.ne),
            Face(self.ne, self.nw)
        ]

        # Surface of the cell adding up the areas of the two composing
        # triangles (nw, sw, se) and (se, ne nw)
        self.volume = np.linalg.norm(
            np.cross(
                self.sw - self.nw,
                self.se - self.sw
            )
        )/2

        self.volume = self.volume + np.linalg.norm(
            np.cross(
                self.se - self.ne,
                self.ne - self.nw
            )
        )/2

        self.centroid = (
            (nw[0] + sw[0] + se[0] + ne[0])/4,
            (nw[1] + sw[1] + se[1] + ne[1])/4,
        )

        self.i = i
        self.j = j

        super().__init__(value)

    def __repr__(self):
        return f'Cell(' \
               f'{self.nw}, {self.sw}, {self.se}, {self.ne})'

    def __iter__(self):
        all_neigh = [self.w, self.s, self.e, self.n]
        filtered = [neigh for neigh in all_neigh if neigh is not None]

        return iter(filtered)

    def update(self):
        self.value = self.new

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w):
        if w is not None:
            self._w = NeighbourCell(w, self.faces[0])
        else:
            self._w = None

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, s):
        if s is not None:
            self._s = NeighbourCell(s, self.faces[1])
        else:
            self._s = None

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, e):
        if e is not None:
            self._e = NeighbourCell(e, self.faces[2])
        else:
            self._e = None

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        if n is not None:
            self._n = NeighbourCell(n, self.faces[3])
        else:
            self._n = None

    def points(self):
        return [self.nw, self.sw, self.se, self.ne]

    def plot(self):
        for face in self.faces:
            face.plot()
        plt.plot(self.centroid[0], self.centroid[1], 'kx', ms=5)


class Face:
    def __init__(self, a, b):
        self._a = np.array(a)
        self._b = np.array(b)

        # Relative position vector
        r = self._b - self._a

        # Normal vector
        self.normal = np.array([r[1], -r[0]])

        # Normalize
        self.normal = self.normal/np.linalg.norm(self.normal)

        self.surface = np.linalg.norm(r)

    def plot(self):
        plt.plot([self._a[0], self._b[0]], [self._a[1], self._b[1]], 'k-')


class NeighbourCell:
    def __init__(self, cell: Cell, face: Face):
        self.face = face
        self.cell = cell

    @property
    def value(self):
        return self.cell.value
