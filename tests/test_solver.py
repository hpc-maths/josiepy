import numpy as np
import pytest

from josie.solver.solver import Solver
from josie.solver.state import StateTemplate, State
from josie.mesh.cell import NeighbourCell, GhostCell
from josie.solver.problem import Problem


@pytest.fixture
def advection(mesh):

    class Advection(Problem):
        Q = StateTemplate("u")
        V = np.array([1, 0])

        @classmethod
        def flux(cls, Q: State) -> State:
            return cls.V*Q

    yield Solver(mesh, Advection)


@pytest.fixture
def init_fun(advection):
    def init_fun(cell):
        xc, yc = cell.centroid

        if xc > 0.45:
            return advection.problem.Q(1)
        else:
            return advection.problem.Q(0)

    yield init_fun


def test_init(advection, init_fun):

    advection.init(init_fun)

    def assert_init(cell):
        if isinstance(cell, GhostCell):
            return True

        xc, _ = cell.centroid

        if xc > 0.45:
            assert np.array_equal(cell.value, advection.problem.Q(1))
        else:
            assert np.array_equal(cell.value, advection.problem.Q(0))

    for cell in advection.mesh.cells.ravel():
        assert_init(cell)

    for left_cell in advection.mesh.cells[0, :]:
        assert isinstance(left_cell.w, NeighbourCell)

    for btm_cell in advection.mesh.cells[:, 0]:
        assert isinstance(left_cell.s, NeighbourCell)

    for right_cell in advection.mesh.cells[-1, :]:
        assert isinstance(left_cell.e, NeighbourCell)

    for top_cell in advection.mesh.cells[-1, :]:
        assert isinstance(left_cell.n, NeighbourCell)


def test_neigh_state(advection, init_fun):
    advection.init(init_fun)

    cell = advection.mesh.cells[0, 0]
    east_neigh = advection.mesh.cells[1, 0]

    assert cell.e.value == east_neigh.value

    east_neigh.value = advection.problem.Q(14)

    assert cell.e.value == east_neigh.value


def test_advection(advection, init_fun):
    advection.init(init_fun)

    def upwind(cell):
        Q = advection.problem.Q(0)
        for neigh in cell:
            norm = neigh.face.normal
            flux = advection.problem.flux
            S = neigh.face.surface

            un = advection.problem.V.dot(neigh.face.normal)

            if un >= 0:
                Q = Q + flux(cell.value).dot(norm)*S
            else:
                Q = Q + flux(neigh.value).dot(norm)*S

        return Q

    advection.solve(2, 0.01, upwind)
