import numpy as np

from typing import Callable

from tvtk.api import tvtk, write_data
# from mayavi import mlab

from josie.mesh import Mesh
from josie.mesh.cell import Cell, GhostCell
from .state import State
from .problem import Problem


class Solver:
    def __init__(self, mesh: Mesh, problem: Problem):
        self.mesh = mesh
        self.problem = problem

    def init(self, init_fun: Callable[[Cell], State]):
        num_cells_x = self.mesh.num_cells_x
        num_cells_y = self.mesh.num_cells_y
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                c = self.mesh.cells[i, j]
                c.value = init_fun(c)

                # Add neighbours and handle BCs
                try:
                    neigh = self.mesh.cells[i-1, j]
                    neigh.value = init_fun(neigh)
                    c.w = neigh
                except IndexError:
                    # Left BC
                    c.w = GhostCell(self.mesh.left.bc(self.mesh, c))

                try:
                    neigh = self.mesh.cells[i, j-1]
                    neigh.value = init_fun(neigh)
                    c.s = neigh
                except IndexError:
                    # Bottom BC
                    c.s = GhostCell(self.mesh.bottom.bc(self.mesh, c))

                try:
                    neigh = self.mesh.cells[i+1, j]
                    neigh.value = init_fun(neigh)
                    c.e = neigh
                except IndexError:
                    # Right BC
                    c.e = GhostCell(self.mesh.right.bc(self.mesh, c))

                try:
                    neigh = self.mesh.cells[i, j+1]
                    neigh.value = init_fun(neigh)
                    c.n = neigh
                except IndexError:
                    # Top BC
                    c.n = GhostCell(self.mesh.top.bc(self.mesh, c))

    def solve(self, final_time, dt, scheme):
        t = 0
        i = 0

        while t < final_time:
            self.save(f't_{i:02d}.vtk')
            t = t+dt
            i = i+1

            for cell in self.mesh.cells.ravel():
                cell.update()
                fluxes = scheme(cell)
                cell.value = cell.old - dt/cell.volume*fluxes

    def save(self, filename):
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

        write_data(sgrid, filename)

    # def plot(self):
        # mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
        #             figure=sgrid.class_name[3:])
        # surf = mlab.pipeline.surface(sgrid, opacity=0.1)
        # mlab.pipeline.surface(mlab.pipeline.extract_edges(surf),
        #                       color=(0, 0, 0), )

        # print(mlab.view())
        # mlab.view(azimuth=0, elevation=0)
        # mlab.show()
