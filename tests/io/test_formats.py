import numpy as np
import pytest

from josie.io.write.formats import MemoryWriter, NoopWriter, XDMFWriter
from josie.solver.state import StateTemplate


@pytest.fixture
def solver(mesh, mocker):
    solver = mocker.Mock()
    solver.values = np.zeros(mesh.num_cells_x * mesh.num_cells_y)
    solver.mesh = mesh
    solver.scheme.CFL = mocker.Mock(return_value=0.1)
    solver.Q = StateTemplate("u")

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


def test_memory(solver, strategy):
    writer = MemoryWriter(strategy, solver, final_time=1.0, CFL=0.5)

    writer.solve()

    assert len(writer.data) == 11


def test_xdmf(solver, strategy, mesh, tmp_path, mocker):

    filename = tmp_path / "test.xdmf"

    writer = XDMFWriter(filename, strategy, solver, final_time=1.0, CFL=0.5)

    writer.solve()

    assert filename.stat().st_size != 0
