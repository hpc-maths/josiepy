# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from josie.io.write.strategy import (
    NoopStrategy,
    IterationStrategy,
    TimeStrategy,
)


@pytest.fixture
def solver(mocker):
    solver = mocker.Mock()
    solver.scheme.CFL = mocker.Mock(return_value=0.01)

    yield solver


def test_noop(solver):
    strategy = NoopStrategy()

    ts = np.linspace(0, 100, 10)
    dt = solver.scheme.CFL()

    for t in ts:
        # Check that the dt returned by the writer is potentially equal or
        # smaller to the physical one given by the CFL condition
        assert strategy.check_write(t, dt, solver) <= solver.scheme.CFL()

        # NoopWriter never writes
        assert strategy.should_write is False


def test_time(solver, tol):
    strategy = TimeStrategy(dt_save=0.1)

    dt = solver.scheme.CFL()
    ts = np.arange(0, 1, dt)

    count = 0
    for t in ts:
        dt_new = strategy.check_write(t, dt, solver)

        # Check that the dt returned by the writer is potentially equal or
        # smaller to the physical one given by the CFL condition
        assert dt_new <= solver.scheme.CFL()

        if strategy.should_write:
            count += 1

    # Correct number of writes
    assert count == 11


def test_iteration(solver):
    strategy = IterationStrategy(n=10)

    ts = np.linspace(0, 1, 10)
    dt = solver.scheme.CFL()

    count = 0
    for t in ts:
        dt_new = strategy.check_write(t, dt, solver)

        # Check that the dt returned by the writer is potentially equal or
        # smaller to the physical one given by the CFL condition
        assert dt_new <= solver.scheme.CFL()

        if strategy.should_write:
            count += 1

    assert count == 10
