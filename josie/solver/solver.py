from typing import Callable, Type

from josie.mesh import Mesh
from .state import State


class Solver:
    def __init__(self, mesh: Mesh, state_cls: Type[State]):
        self.mesh = mesh

    def init(self, init_fun: Callable):
        for cell in self.mesh.cells:
            cell.new = init_fun(cell.centroid)
