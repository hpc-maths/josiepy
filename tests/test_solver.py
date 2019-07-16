import numpy as np

from josie.solver.solver import Solver
from josie.solver.state import StateTemplate
from josie.mesh.cell import NeighbourCell


def test_init(mesh):
    Q = StateTemplate("u")
    solver = Solver(mesh, Q)

    def init_zero(cell):
        return Q(0)

    solver.init(init_zero)

    for cell in solver.mesh.cells.ravel():
        assert np.array_equal(cell.new, Q(0))

    for left_cell in solver.mesh.cells[0, :]:
        assert isinstance(left_cell.w, NeighbourCell)

    for btm_cell in solver.mesh.cells[:, 0]:
        assert isinstance(left_cell.s, NeighbourCell)

    for right_cell in solver.mesh.cells[-1, :]:
        assert isinstance(left_cell.e, NeighbourCell)

    for top_cell in solver.mesh.cells[-1, :]:
        assert isinstance(left_cell.n, NeighbourCell)
