import matplotlib.pyplot as plt
import numpy as np
import pytest

from collections import namedtuple
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from josie.solver.euler.eos import (
    PerfectGas,
    # StiffenedGas,
)
from josie.geom import Line
from josie.bc import Dirichlet
from josie.mesh import Mesh, SimpleCell
from josie.solver.scheme.time import ExplicitEuler
from josie.solver.twophase.eos import TwoPhaseEOS
from josie.solver.twophase.closure import Closure, Classical
from josie.solver.twophase.schemes import Rusanov, Upwind
from josie.solver.twophase.state import Phases, PhasePair, Q
from josie.solver.twophase.solver import TwoPhaseSolver


class ToroScheme(ExplicitEuler, Rusanov):
    pass


# These class are used to store the left/right state for each phase

RiemannState = namedtuple("RiemannState", ["rho", "U", "V", "p", "eos"])


@dataclass
class RiemannBCState:
    alpha: float
    state: PhasePair


@dataclass
class RiemannProblem:
    left: RiemannBCState
    right: RiemannBCState
    closure: Closure
    CFL: float


riemann_states = [
    RiemannProblem(
        left=RiemannBCState(
            alpha=0,
            state=PhasePair(
                phase1=RiemannState(
                    rho=1, U=0, V=0, p=1, eos=PerfectGas(gamma=1.4)
                ),
                phase2=RiemannState(
                    rho=1, U=0, V=0, p=1, eos=PerfectGas(gamma=1.4)
                ),
            ),
        ),
        right=RiemannBCState(
            alpha=0,
            state=PhasePair(
                phase1=RiemannState(
                    rho=0.125, U=0, V=0, p=0.1, eos=PerfectGas(gamma=1.4)
                ),
                phase2=RiemannState(
                    rho=0.125, U=0, V=0, p=0.1, eos=PerfectGas(gamma=1.4)
                ),
            ),
        ),
        closure=Classical(),
        CFL=0.5,
    ),
    RiemannProblem(
        left=RiemannBCState(
            alpha=0.8,
            state=PhasePair(
                phase1=RiemannState(
                    rho=1, U=0, V=0, p=1, eos=PerfectGas(gamma=1.4)
                ),
                phase2=RiemannState(
                    rho=0.2, U=0, V=0, p=0.3, eos=PerfectGas(gamma=1.4)
                ),
            ),
        ),
        right=RiemannBCState(
            alpha=0.3,
            state=PhasePair(
                phase1=RiemannState(
                    rho=1, U=0, V=0, p=1, eos=PerfectGas(gamma=1.4)
                ),
                phase2=RiemannState(
                    rho=1, U=0, V=0, p=1, eos=PerfectGas(gamma=1.4)
                ),
            ),
        ),
        closure=Classical(),
        CFL=0.1,
    ),
]


def set_bc_state(bc_state: RiemannBCState):
    fields = Q.fields
    state_array: Q = np.zeros(len(fields)).view(Q)

    alpha = bc_state.alpha
    state_array[..., fields.alpha] = alpha

    for phase, phase_data in bc_state.state.items():
        eos: PerfectGas = phase_data.eos

        rho = phase_data.rho
        U = phase_data.U
        rhoU = rho * U
        V = phase_data.V
        rhoV = rho * V
        p = phase_data.p

        rhoe = eos.rhoe(rho, p)
        E = rhoe / rho + 0.5 * (U ** 2 + V ** 2)
        rhoE = rho * E
        c = eos.sound_velocity(rho, p)

        state_array.set_phase(
            phase, np.asarray((rho, rhoU, rhoV, rhoE, rhoe, U, V, p, c))
        )

    return state_array


def plot_func(data, lines, axes, fields_to_plot):
    x = data[0]
    values = data[1]

    # Alpha
    line = lines[0]
    line.set_data(x, values[..., Q.fields.alpha])
    ax = axes[0]
    ax.relim()
    ax.autoscale_view()

    for i, field in enumerate(fields_to_plot, 1):
        line = lines[i]
        line.set_data(x, values[..., field])
        ax = axes[i]
        ax.relim()
        ax.autoscale_view()


@pytest.mark.parametrize("riemann_problem", riemann_states)
def test_toro(riemann_problem: RiemannProblem, plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    # BC
    Q_left: Q = set_bc_state(riemann_problem.left)
    Q_right: Q = set_bc_state(riemann_problem.right)

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(500, 1)
    mesh.generate()

    def init_fun(solver: TwoPhaseSolver):
        xc = solver.mesh.centroids[:, :, 0]

        solver.values[np.where(xc > 0.5), :, :] = Q_right
        solver.values[np.where(xc <= 0.5), :, :] = Q_left

    eos = TwoPhaseEOS(
        phase1=riemann_problem.left.state.phase1.eos,
        phase2=riemann_problem.left.state.phase2.eos,
    )
    closure = riemann_problem.closure

    scheme = ToroScheme(eos, closure)
    solver = TwoPhaseSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = 0.25
    t = 0
    CFL = riemann_problem.CFL

    # Initialize stuff per each phase, plot only conservative stuff
    if plot:
        fields = Q.fields
        fields_to_plot = [
            fields.rho1,
            fields.rho2,
            fields.U1,
            fields.U2,
            fields.p1,
            fields.p2,
        ]
        num_fields = len(fields_to_plot) // 2

        time_series = []
        artists = []
        axes = []

        x = solver.mesh.centroids[..., 0]
        x = x.reshape(x.size)

        fig = plt.figure()

        # First plot alpha
        num_fields += 1
        gs = GridSpec(num_fields, 2)
        ax: plt.Axes = fig.add_subplot(gs[0, :])
        field_value = solver.values[..., fields.alpha]
        (line,) = ax.plot(x, field_value, label=r"$\alpha$")
        ax.legend(loc="best")
        artists.append(line)
        axes.append(ax)

        for i, field in enumerate(fields_to_plot, 2):
            # Indices in the plot grid
            idx, idy = np.unravel_index(i, (num_fields, 2))
            ax: plt.Axes = fig.add_subplot(gs[idx, idy])
            field_value = solver.values[..., field]
            (line,) = ax.plot(x, field_value, label=field.name)
            ax.legend(loc="best")
            artists.append(line)
            axes.append(ax)

    while t <= final_time:
        if plot:
            x = solver.mesh.centroids[..., 0]
            x = x.reshape(x.size)

            time_series.append((x, np.copy(solver.values).view(Q)))

        dt = scheme.CFL(
            solver.values,
            solver.mesh.volumes,
            solver.mesh.normals,
            solver.mesh.surfaces,
            CFL,
        )
        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        fig.tight_layout()
        _ = FuncAnimation(
            fig,
            plot_func,
            [(data[0], data[1]) for data in time_series],
            fargs=(artists, axes, fields_to_plot),
        )
        plt.show()
        plt.close()
