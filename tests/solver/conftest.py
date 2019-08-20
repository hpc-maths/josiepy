import numpy as np
import pytest

from josie.solver.state import StateTemplate
from josie.solver.solver import Solver


@pytest.fixture
def Q():
    """ Simple scalar state for easy check """
    Q = StateTemplate("u")

    yield Q


@pytest.fixture
def init_fun(Q):
    """ Init a step in the state """

    def init_fun(solver: Solver):
        xc = solver.mesh.centroids[:, :, 0]

        solver.values[np.where(xc >= 0.45), :, :] = Q(1)
        solver.values[np.where(xc < 0.45), :, :] = Q(0)

    yield init_fun


@pytest.fixture
def solver(mesh, Q, init_fun):
    """ A dummy solver instance with initiated state """

    solver = Solver(mesh, Q)
    solver.init(init_fun)

    yield solver
