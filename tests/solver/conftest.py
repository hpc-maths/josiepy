import numpy as np
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
    def init_fun(solver: Solver):
        xc = solver.mesh.centroids[:, :, 0]

        solver.values[np.where(xc >= 0.45), :, :] = problem.Q(1)
        solver.values[np.where(xc < 0.45), :, :] = problem.Q(0)

    yield init_fun


@pytest.fixture
def solver(mesh, problem, init_fun):
    """ A dummy solver instance with initiated state """

    solver = Solver(mesh, problem.Q)
    solver.init(init_fun)

    yield solver
