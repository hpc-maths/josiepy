import numpy as np
import pytest

from josie.io.write.strategy import (
    NoopStrategy,
    IterationStrategy,
    TimeStrategy,
)


@pytest.fixture
def solver(mocker):
    yield mocker.Mock()


def test_noop(solver):
    strategy = NoopStrategy()

    ts = np.linspace(0, 100, 10)
    dts = np.linspace(0.01, 0.1, 10)

    for t in ts:
        for dt in dts:
            strategy.check_write(t, dt, solver)
            assert strategy.should_write is False


def test_time(solver, tol):
    strategy = TimeStrategy(dt_save=0.1)

    dt = 0.1
    ts = np.arange(0, 1, dt)

    count = 0
    for t in ts:
        dt_new = strategy.check_write(t, dt, solver)
        if strategy.should_write:
            count += 1
            assert dt - dt_new < tol

    assert count == 10


def test_iteration(solver):
    strategy = IterationStrategy(n=10)

    ts = np.linspace(0, 1, 10)
    dt = 0.1
    count = 0
    for t in ts:
        strategy.check_write(t, dt, solver)
        if strategy.should_write:
            count += 1

    assert count == 10
