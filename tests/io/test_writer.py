import pytest

from josie.io.write.writer import MemoryWriter, NoopWriter, XDMFWriter
from josie.state import StateTemplate


@pytest.fixture
def solver(mesh, mocker):
    solver = mocker.Mock()
    # Allow pickling
    solver.__reduce__ = lambda self: (mocker.Mock, ())
    solver.t = 0
    solver.mesh = mesh
    solver.mesh.cells._values = mocker.MagicMock()
    solver.scheme.CFL = mocker.Mock(return_value=0.1)
    solver.Q = StateTemplate("u")

    def step_func(self, dt):
        self.t += dt

    solver.step = step_func.__get__(solver)

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

    assert solver.t >= 1.0


def test_memory(solver, strategy):
    writer = MemoryWriter(strategy, solver, final_time=1.0, CFL=0.5)

    writer.solve()

    assert len(writer.data) == 11


def test_xdmf(solver, strategy, mesh, tmp_path, mocker):

    filename = tmp_path / "test.xdmf"

    writer = XDMFWriter(filename, strategy, solver, final_time=1.0, CFL=0.5)

    writer.solve()

    assert filename.stat().st_size != 0
