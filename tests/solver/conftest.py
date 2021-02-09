import numpy as np
import pytest

from josie.state import StateTemplate
from josie.solver import Solver
from josie.mesh.cellset import MeshCellSet


@pytest.fixture
def Q():
    """ Simple scalar state for easy check """
    Q = StateTemplate("u")

    yield Q


@pytest.fixture
def init_fun(Q):
    """ Init a step in the state """

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]

        xc_r = np.where(xc >= 0.45)
        xc_l = np.where(xc < 0.45)

        cells.values[xc_r[0], xc_r[1], :] = Q(1)
        cells.values[xc_l[0], xc_l[1], :] = Q(0)

    yield init_fun


@pytest.fixture
def solver(mocker, mesh, Q, init_fun):
    """ A dummy solver instance with initiated state """

    scheme = mocker.Mock()

    solver = Solver(mesh, Q, scheme)
    solver.init(init_fun)

    yield solver
