import inspect
import numpy as np
import pytest

import matplotlib.pyplot as plt


import josie.general.schemes.time as time_schemes

from josie.ode import OdeSolver
from josie.solver.state import State, StateTemplate

Q = StateTemplate("x", "v")


class OscillatorRHS:
    def __init__(self, k: float, m: float):
        self.k = k
        self.m = m

    def __call__(self, values) -> State:
        fields = values.fields
        source = values.copy()

        source[..., fields.x] = values[..., fields.v]
        source[..., fields.v] = -self.k / self.m * values[..., fields.x]

        return source


def analytical_solution(t, Q0, omega):
    # Q0 initial conditions
    x0 = Q0[..., Q0.fields.x]
    v0 = Q0[..., Q0.fields.v]

    x_t = v0 / omega * np.sin(omega * t) + x0 * np.cos(omega * t)
    v_t = v0 * np.cos(omega * t) - x0 * omega * np.sin(omega * t)

    Q_t = np.concatenate(
        (np.atleast_1d(x_t), np.atleast_1d(v_t)), axis=-1
    ).view(Q)

    return Q_t


def run_oscillator(
    mesh, init_fun, Q0, TimeScheme, k, m, dt, final_time
) -> float:
    """Testing against the analytical solution of an harmonic oscillator
    without damping"""

    solver = OdeSolver(Q, TimeScheme, OscillatorRHS(k, m))
    solver.init(init_fun)

    ts = []

    xs = []
    xs_exact = []
    err_xs = []

    vs = []
    vs_exact = []
    err_vs = []

    while solver.t <= final_time:
        values = solver.mesh.cells.values

        Q_exact = analytical_solution(solver.t, Q0, np.sqrt(k / m))
        x_exact = Q_exact[..., Q.fields.x].item()
        v_exact = Q_exact[..., Q.fields.v].item()

        x = values[..., Q.fields.x].reshape(1).copy().item()
        v = values[..., Q.fields.v].reshape(1).copy().item()

        # Error in position
        err_x = np.abs(x - x_exact)
        err_v = np.abs(v - v_exact)

        ts.append(solver.t)

        err_xs.append(err_x)
        err_vs.append(err_v)
        xs.append(x)
        vs.append(v)
        xs_exact.append(x_exact)
        vs_exact.append(v_exact)

        solver.step(dt)

        solver.t += dt

        # print(f"Time: {solver.t}, dt: {dt}")

    return xs, xs_exact, err_xs, vs, vs_exact, err_vs, ts


@pytest.fixture
def Q0():
    """ Initial state """

    return np.array([1, 0]).view(Q)


@pytest.fixture()
def init_fun(Q0):
    """ Initial position at 1 and 0 velocity """

    fields = Q.fields

    def _init_fun(cells):
        cells.values[..., fields.x] = Q0[..., fields.x]
        cells.values[..., fields.v] = Q0[..., fields.v]

    yield _init_fun


@pytest.fixture(
    params=[
        member[1]
        for member in inspect.getmembers(time_schemes, inspect.isclass)
    ]
)
def TimeScheme(request):
    yield request.param


def test_oscillator(mesh, init_fun, Q0, TimeScheme, plot):
    """Testing against the analytical solution of an harmonic oscillator
    without damping"""

    k = 10
    m = 1
    dt = 0.001
    final_time = 2

    xs, xs_exact, err_xs, vs, vs_exact, err_vs, ts = run_oscillator(
        mesh, init_fun, Q0, TimeScheme, k, m, dt, final_time
    )

    # Assert the average error is less then 1e-3
    # This is an euristics
    total_error_x = np.sum(err_xs) / len(err_xs)
    assert total_error_x < 5e-3

    if plot:
        fig = plt.figure()

        x_ax = fig.add_subplot(2, 2, 1)
        x_ax.plot(ts, xs, "x", label=r"$x(t)$ numerical")
        x_ax.plot(ts, xs_exact, "-", label=r"$x(t)$ analytical")
        x_ax.set_xlabel(r"$t$")
        x_ax.set_ylabel(r"$x(t)$")
        x_ax.legend()
        x_ax.grid()

        err_x_ax = fig.add_subplot(2, 2, 3)
        err_x_ax.plot(ts, err_xs, "-")
        err_x_ax.set_xlabel(r"$t$")
        err_x_ax.set_ylabel(r"$e_x(t)$")
        err_x_ax.set_title("Absolute $L_1$ error for the position")
        err_x_ax.grid()

        v_ax = fig.add_subplot(2, 2, 2)
        v_ax.plot(ts, vs, "x", label=r"$v(t)$ numerical")
        v_ax.plot(ts, vs_exact, "-", label=r"$v(t)$ analytical")
        v_ax.set_xlabel(r"$t$")
        v_ax.set_ylabel(r"$v(t)$")
        v_ax.legend()
        v_ax.grid()

        err_v_ax = fig.add_subplot(2, 2, 4)
        err_v_ax.plot(ts, err_vs, "-")
        err_v_ax.set_xlabel(r"$t$")
        err_v_ax.set_ylabel(r"$e_v(t)$")
        err_v_ax.set_title("Relative $L_1$ error for the velocity")
        err_v_ax.grid()

        fig.suptitle(TimeScheme.__name__)
        plt.tight_layout()
        plt.show()


def test_oscillator_order(mesh, init_fun, Q0, TimeScheme, plot):
    """Measure convergence order against the analytical solution of an harmonic
    oscillator without damping"""

    k = 10
    m = 1
    dts = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    final_time = 2
    err_L2s = []

    for dt in dts:
        _, _, err_xs, _, _, err_vs, ts = run_oscillator(
            mesh, init_fun, Q0, TimeScheme, k, m, dt, final_time
        )

        # Compute the L2 discrete error
        err = np.array([err_xs[1:], err_vs[1:]])
        err_2_dt = np.power(err, 2) * (np.array(ts[1:]) - np.array(ts[:-1]))
        err_L2s.append(np.sqrt(np.sum(err_2_dt)))

    order = np.linalg.lstsq(
        np.vstack([np.log(dts), np.ones(len(dts))]).T,
        np.log(err_L2s),
        rcond=None,
    )[0][0]

    # Assert the order
    assert np.abs(order - TimeScheme.time_order) < 0.1

    if plot:
        fig = plt.figure()

        err_ax = fig.add_subplot(1, 1, 1)
        err_ax.plot(dts, err_L2s, "x")
        err_ax.plot(
            dts,
            err_L2s[-1] * np.power(dts / dts[-1], 1),
            "-",
            label=r"order 1",
        )
        err_ax.plot(
            dts,
            err_L2s[-1] * np.power(dts / dts[-1], 2),
            "-",
            label=r"order 2",
        )
        err_ax.plot(
            dts,
            err_L2s[-1] * np.power(dts / dts[-1], 3),
            "-",
            label=r"order 3",
        )
        plt.xscale("log")
        plt.yscale("log")
        err_ax.set_xlabel(r"$dt$")
        err_ax.set_ylabel(r"$L^2$-error")
        err_ax.legend()
        err_ax.grid()

        fig.suptitle(TimeScheme.__name__)
        plt.tight_layout()
        plt.show()
