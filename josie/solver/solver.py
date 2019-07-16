from typing import Callable

from josie.mesh import Mesh
from josie.mesh.cell import Cell, GhostCell
from .state import State, StateTemplate


class Solver:
    def __init__(self, mesh: Mesh, state_template: StateTemplate):
        self.mesh = mesh

    def init(self, init_fun: Callable[[Cell], State]):
        num_cells_x = self.mesh.num_cells_x
        num_cells_y = self.mesh.num_cells_y
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                c = self.mesh.cells[i, j]
                c.new = init_fun(c)

                # Add neighbours and handle BCs
                try:
                    c.w = self.mesh.cells[i-1, j]
                except IndexError:
                    # Left BC
                    c.w = GhostCell(self.mesh.left.bc(self, c))

                try:
                    c.s = self.mesh.cells[i, j-1]
                except IndexError:
                    # Bottom BC
                    c.s = GhostCell(self.mesh.bottom.bc(self, c))

                try:
                    c.e = self.mesh.cells[i+i, j]
                except IndexError:
                    # Right BC
                    c.e = GhostCell(self.mesh.right.bc(self, c))

                try:
                    c.n = self.mesh.cells[i, j+1]
                except IndexError:
                    # Top BC
                    c.n = GhostCell(self.mesh.top.bc(self, c))
