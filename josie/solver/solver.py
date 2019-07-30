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

import numpy as np

from typing import Callable, TYPE_CHECKING

from josie.exceptions import InvalidMesh

from .problem import Problem
from .state import State

if TYPE_CHECKING:
    from josie.mesh import Mesh
    from josie.mesh.cell import Cell


class Solver:
    def __init__(self, mesh: 'Mesh', problem: Problem):
        self.mesh = mesh
        self.problem = problem

    def init(self, init_fun: Callable[[Cell], State]):
        num_cells_x = self.mesh.num_cells_x
        num_cells_y = self.mesh.num_cells_y

        # First set all the values for the interal cells
        for c in self.mesh.cells.ravel():
            c.value = init_fun(c)

        for i in range(num_cells_x):
            for j in range(num_cells_y):
                c = self.mesh.cells[i, j]

                # Left BC
                if i == 0:
                    if self.mesh.left.bc is not None:
                        c.w = self.mesh.left.bc(self.mesh, c)
                        # Right neighbour
                        neigh = self.mesh.cells[i+1, j]
                        c.e = neigh
                    else:
                        c.e = None
                        c.w = None

                # Right BC
                elif i == (num_cells_x - 1):
                    if self.mesh.right.bc is not None:
                        c.e = self.mesh.right.bc(self.mesh, c)
                        # Left neighbour
                        neigh = self.mesh.cells[i-1, j]
                        c.w = neigh
                    else:
                        # If we hit this, it means a 1D mesh in y direction has
                        # been defined but with more than one row of cells in
                        # the x direction. This is not supported.
                        raise InvalidMesh(
                            "You defined a 1D mesh in the y direction since "
                            "the top and bottom BC are `None`. But you "
                            "initialized a mesh with more that 1 cell in "
                            "the y direction")

                # Normal Cell
                else:
                    # Left neighbour
                    neigh = self.mesh.cells[i-1, j]
                    c.w = neigh

                    # Right neighbour
                    neigh = self.mesh.cells[i+1, j]
                    c.e = neigh

                # Bottom BC
                if j == 0:
                    if self.mesh.bottom.bc is not None:
                        c.s = self.mesh.bottom.bc(self.mesh, c)
                        # Top neighbour
                        neigh = self.mesh.cells[i, j+1]
                        c.n = neigh
                    else:
                        c.n = None
                        c.s = None

                # Top BC
                elif j == (num_cells_y - 1):
                    if self.mesh.top.bc is not None:
                        c.n = self.mesh.top.bc(self.mesh, c)
                        # Bottom neighbour
                        neigh = self.mesh.cells[i, j-1]
                        c.s = neigh
                    else:
                        # If we hit this, it means a 1D mesh in x direction has
                        # been defined but with more than one row of cells in
                        # the y direction. This is not supported.
                        raise InvalidMesh(
                            "You defined a 1D mesh in the x direction since "
                            "the top and bottom BC are `None`. But you "
                            "initialized a mesh with more that 1 cell in "
                            "the y direction")

                # Normal Cell
                else:

                    # Top neighbour
                    neigh = self.mesh.cells[i, j+1]
                    c.n = neigh

                    # Bottom neighbour
                    neigh = self.mesh.cells[i, j-1]
                    c.s = neigh

    def step(self, dt, scheme):
        for cell in self.mesh.cells.ravel():
            fluxes = scheme(cell)
            cell.new = cell.value - dt/cell.volume*fluxes

        for cell in self.mesh.cells.ravel():
            cell.update()

    def solve(self, final_time, dt, scheme, animate=False, write=False):
        if animate:
            self._init_show()
        t = 0
        i = 0

        while t < final_time:
            t = t+dt
            i = i+1

            if animate:
                self.animate()

            self.step(dt, scheme)

            if write:
                self.save(f't_{i:02d}.vtk')

    def _to_mayavi(self):
        from tvtk.api import tvtk
        # Rearrange points
        points = np.vstack((self.mesh._x.ravel(), self.mesh._y.ravel())).T
        points = np.pad(points, ((0, 0), (0, 1)), 'constant')

        sgrid = tvtk.StructuredGrid(
            dimensions=(self.mesh._num_xi, self.mesh._num_eta, 1)
        )

        sgrid.points = points

        cell_data = np.empty((len(self.mesh.cells.ravel()),
                             len(self.problem.Q.fields)))

        for i, cell in enumerate(self.mesh.cells.ravel()):
            cell_data[i, :] = cell.value

        sgrid.cell_data.scalars = cell_data

        return sgrid

    def save(self, filename):
        from tvtk.api import write_data
        sgrid = self._to_mayavi()
        write_data(sgrid, filename)

    def _init_show(self):
        from mayavi import mlab
        sgrid = self._to_mayavi()

        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                    figure=sgrid.class_name[3:])
        surf = mlab.pipeline.surface(sgrid, opacity=0.1)
        mlab.pipeline.surface(mlab.pipeline.extract_edges(surf),
                              color=(0, 0, 0), )

        mlab.view(azimuth=0, elevation=0)

        self._surf = surf
        self._sgrid = sgrid

    def animate(self):
        cell_data = np.empty((len(self.mesh.cells.ravel()),
                             len(self.problem.Q.fields)))

        for i, cell in enumerate(self.mesh.cells.ravel()):
            cell_data[i, :] = cell.value

        self._sgrid.cell_data.scalars = cell_data
        self._sgrid.modified()
        yield
