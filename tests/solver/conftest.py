import pytest

from josie.solver.problem import Problem
from josie.solver.state import StateTemplate
from josie.solver.solver import Solver


@pytest.fixture
def problem():
    """ Just defining a dummy problem to test initialization and stuff """

    class Dummy(Problem):
        Q = StateTemplate("u")

        @classmethod
        def flux(cls, Q):
            pass

    yield Dummy


@pytest.fixture
def init_fun(problem):
    """ Init a step in the state """
    def init_fun(cell):
        xc, yc = cell.centroid

        if xc > 0.45:
            return problem.Q(1)
        else:
            return problem.Q(0)

    yield init_fun


@pytest.fixture
def solver(mesh, problem, init_fun):
    """ A dummy solver instance with initiated state """

    solver = Solver(mesh, problem)
    solver.init(init_fun)

    yield solver
