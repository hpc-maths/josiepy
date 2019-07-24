import numpy as np
import pytest

from josie.solver.problem import Problem
from josie.solver.solver import Solver
from josie.solver.state import StateTemplate, State


class Advection(Problem):
    # Scalar advection
    Q = StateTemplate('u')

    # Advection velocity in x-direction
    V = np.array([1, 0])

    @classmethod
    def flux(cls, Q: State) -> np.ndarray:
        return cls.V*Q


@pytest.fixture
def solver(mesh, init_fun):
    solver = Solver(mesh, Advection)
    solver.init(init_fun)

    yield solver
