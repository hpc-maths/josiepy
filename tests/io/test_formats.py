import numpy as np
import pytest

from josie.io.write.formats import NoopWriter, XDMFWriter
from josie.solver.state import StateTemplate


@pytest.fixture
def solver(mocker):
    solver = mocker.Mock()
    solver.scheme.CFL = mocker.Mock(return_value=0.1)
    solver.values = mocker.Mock(return_value=np.zeros(10))

    yield solver


@pytest.fixture
def strategy(mocker):
    strategy = mocker.Mock()
    strategy.should_write = mocker.Mock(return_value=True)
    strategy.check_write = mocker.Mock(return_value=0.1)

    yield strategy


def test_noop(solver):
    writer = NoopWriter(solver, final_time=1.0, CFL=0.5)

    writer.solve()

    assert solver.step.call_count == 11


def test_xdmf(strategy, mesh, tmp_path, mocker):
    solver = mocker.Mock()
    solver.values = np.zeros(mesh.num_cells_x * mesh.num_cells_y)
    solver.mesh = mesh
    solver.Q = StateTemplate("u")

    filename = tmp_path / "test.xdmf"

    writer = XDMFWriter(filename, strategy, solver, final_time=1.0, CFL=0.5)

    writer.solve()

    assert filename.stat().st_size != 0
