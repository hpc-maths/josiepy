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

""" This module contains the primitives related to the mesh generation """

import numpy as np

from .cell import Cell, GhostCell


class Mesh:
    """ This class handles the mesh generation over a domain.

    Parameters:
        left: The left BoundaryCurve
        bottom: The bottom BoundaryCurve
        right: The right BoundaryCurve
        top: The right BoundaryCurve

    """

    def __init__(self, left, bottom, right, top):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top

    def interpolate(self, num_xi, num_eta):
        """ This methods generates the mesh within the four given
        BoundaryCurve using Transfinite Interpolation

        Args:
            num_xi: Number of elements in xi-direction
            num_eta: Number of elements in eta-direction
        """
        self._num_xi = num_xi
        self._num_eta = num_eta

        xis = np.linspace(0, 1, num_xi)
        etas = np.linspace(0, 1, num_eta)

        x = np.empty((len(xis), len(etas)))
        y = np.empty((len(xis), len(etas)))

        for i, xi in enumerate(xis):
            for j, eta in enumerate(etas):
                xl, yl = self.left(eta)
                xr, yr = self.right(eta)
                xb, yb = self.bottom(xi)
                xt, yt = self.top(xi)
                xb0, yb0 = self.bottom(0)
                xb1, yb1 = self.bottom(1)
                xt0, yt0 = self.top(0)
                xt1, yt1 = self.top(1)

                x[i, j] = \
                    (1-xi)*xl + xi*xr + \
                    (1-eta)*xb + eta*xt - \
                    (1-xi)*(1-eta)*xb0 - (1-xi)*eta*xt0 - \
                    (1-eta)*xi*xb1 - xi*eta*xt1

                y[i, j] = \
                    (1-xi)*yl + xi*yr + \
                    (1-eta)*yb + eta*yt - \
                    (1-xi)*(1-eta)*yb0 - (1-xi)*eta*yt0 - \
                    (1-eta)*xi*yb1 - xi*eta*yt1

        self._x = x
        self._y = y

    def generate(self):
        """ This method builds the connectivity """
        num_cells_x = self.num_xi-1
        num_cells_y = self.num_eta-1
        cells = np.empty((num_cells_x, num_cells_y), dtype=object)

        for i in range(num_cells_x):
            for j in range(num_cells_y):
                cells[i, j] = Cell(
                    (self._x[i], self._y[i]),
                    (self._x[i+1], self._y[i]),
                    (self._x[i+1], self._y[i+1]),
                    (self._x[i], self._y[i+1]),
                    i,
                    j
                )

        for i in range(num_cells_x):
            for j in range(num_cells_y):
                c = cells[i, j]

                # Add neighbours and handle BCs
                try:
                    c.w = cells[i-1, j]
                except KeyError:
                    # Left BC
                    c.w = GhostCell(self.left.bc(self, c))

                try:
                    c.s = cells[i, j-1]
                except KeyError:
                    # Bottom BC
                    c.s = GhostCell(self.bottom.bc(self, c))

                try:
                    c.e = cells[i+i, j]
                except KeyError:
                    # Right BC
                    c.e = GhostCell(self.right.bc(self, c))

                try:
                    c.n = cells[i, j+1]
                except KeyError:
                    # Top BC
                    c.n = GhostCell(self.top.bc(self, c))

            self.cells = cells
