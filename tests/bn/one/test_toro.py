# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import matplotlib.pyplot as plt
import numpy as np
import pytest

import josie.general.schemes.time as time_schemes
from josie.general.schemes.space.godunov import Godunov

from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from josie.euler.eos import (
    PerfectGas,
    StiffenedGas,
)
from josie.bc import Dirichlet
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.scheme import Scheme
from josie.scheme.convective import ConvectiveScheme
from josie.scheme.nonconservative import NonConservativeScheme
from josie.twofluid.state import PhasePair
from josie.bn.closure import Classical
from josie.bn.eos import TwoPhaseEOS
from josie.bn.schemes import BaerScheme
from josie.bn.solver import BaerSolver
from josie.bn.state import Q


@pytest.fixture(
    params=[member[1] for member in inspect.getmembers(time_schemes, inspect.isclass)],
)
def TimeScheme(request):
    """Fixture that yields all the concrete time schemes implemented in
    :mod:`josie.general.schemes.time`"""
    yield request.param


@pytest.fixture(
    params=[
        subcls
        for subcls in BaerScheme._all_subclasses()
        if issubclass(subcls, NonConservativeScheme)
        and subcls.tag == NonConservativeScheme
    ]
)
def ToroNonConservativeScheme(request):
    """Fixture that yields all the concrete :class:`BaerScheme` that are
    also :class:`NonConservativeScheme`"""
    yield request.param


@pytest.fixture(
    params=[
        subcls
        for subcls in BaerScheme._all_subclasses()
        if issubclass(subcls, ConvectiveScheme) and subcls.tag == ConvectiveScheme
    ]
)
def IntercellFluxScheme(request):
    """Fixture that yields all the concrete :class:`BaerScheme` that are
    also :class:`ConvectiveScheme`"""
    yield request.param


class NoPI(Classical):
    def pI(self, state_array):
        return np.zeros_like(state_array[..., [state_array.fields.p1]])

    def uI(self, state_array):
        uI = np.zeros_like(
            state_array[..., [state_array.fields.U1, state_array.fields.V1]]
        )
        uI[..., 0] = 1

        return uI


# These class are used to store the left/right state for each phase


@dataclass
class RiemannState:
    rho: float
    U: float
    V: float
    p: float


@dataclass
class RiemannBCState:
    alpha: float
    state: PhasePair


@dataclass
class RiemannProblem:
    left: RiemannBCState
    right: RiemannBCState
    scheme: Scheme
    discontinuity_x0: float
    final_time: float
    CFL: float


@pytest.fixture
def riemann_states(IntercellFluxScheme, ToroNonConservativeScheme, TimeScheme):
    class AdvectionOnly(ToroNonConservativeScheme, TimeScheme):
        # Define this to avoid exception of ABC
        def post_step(self, cells):
            pass

        def CFL(
            self,
            cells: MeshCellSet,
            CFL_value,
        ) -> float:
            return 1e-3

    class ToroScheme(
        IntercellFluxScheme,
        ToroNonConservativeScheme,
        TimeScheme,
        Godunov,
    ):
        pass

    tests = [
        # Test #1
        # -------
        # We set a constant value of alpha such that the non-conservative flux
        # doesnt' intervene. The value for the rho, U, V, p values are the same
        # as for the Test #1 in the Euler tests. We need to retrieve the same
        # plots
        RiemannProblem(
            left=RiemannBCState(
                alpha=0.5,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=1,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=1,
                    ),
                ),
            ),
            right=RiemannBCState(
                alpha=0.5,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=0.125,
                        U=0,
                        V=0,
                        p=0.1,
                    ),
                    phase2=RiemannState(
                        rho=0.125,
                        U=0,
                        V=0,
                        p=0.1,
                    ),
                ),
            ),
            scheme=ToroScheme(
                eos=TwoPhaseEOS(
                    phase1=PerfectGas(gamma=1.4), phase2=PerfectGas(gamma=1.4)
                ),
                closure=Classical(),
            ),
            discontinuity_x0=0.5,
            final_time=0.25,
            CFL=0.45,
        ),
        # # Test #2
        # # -------
        # # We set a constant value of rho, U, V, p and just a non-constant
        # # alpha. In addition we set a Closure equation such that pI = 0. This
        # # should just make the non-conservative sheme work only for the alpha
        # # equation
        # RiemannProblem(
        #     left=RiemannBCState(
        #         alpha=0.8,
        #         state=PhasePair(
        #             phase1=RiemannState(
        #                 rho=1,
        #                 U=0,
        #                 V=0,
        #                 p=1,
        #             ),
        #             phase2=RiemannState(
        #                 rho=1,
        #                 U=0,
        #                 V=0,
        #                 p=1,
        #             ),
        #         ),
        #     ),
        #     right=RiemannBCState(
        #         alpha=0.3,
        #         state=PhasePair(
        #             phase1=RiemannState(
        #                 rho=1,
        #                 U=0,
        #                 V=0,
        #                 p=1,
        #             ),
        #             phase2=RiemannState(
        #                 rho=1,
        #                 U=0,
        #                 V=0,
        #                 p=1,
        #             ),
        #         ),
        #     ),
        #     scheme=AdvectionOnly(
        #         eos=TwoPhaseEOS(
        #             phase1=PerfectGas(gamma=1.4), phase2=PerfectGas(gamma=1.4)
        #         ),
        #         closure=NoPI(),
        #     ),
        #     discontinuity_x0=0.5,
        #     final_time=0.25,
        #     CFL=0.45,
        # ),
        # Test # 3
        # -------
        # The same as the test #1 in :cite:`tokareva_toro`
        RiemannProblem(
            left=RiemannBCState(
                alpha=0.8,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=1,
                    ),
                    phase2=RiemannState(
                        rho=0.2,
                        U=0,
                        V=0,
                        p=0.3,
                    ),
                ),
            ),
            right=RiemannBCState(
                alpha=0.3,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=1,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=1,
                    ),
                ),
            ),
            scheme=ToroScheme(
                eos=TwoPhaseEOS(
                    phase1=PerfectGas(gamma=1.4), phase2=PerfectGas(gamma=1.4)
                ),
                closure=Classical(),
            ),
            discontinuity_x0=0.5,
            final_time=0.15,
            CFL=0.45,
        ),
        # Test # 4
        # -------
        # The same as the test #2 in :cite:`tokareva_toro`
        RiemannProblem(
            left=RiemannBCState(
                alpha=0.2,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1900,
                        U=0,
                        V=0,
                        p=10,
                    ),
                    phase2=RiemannState(
                        rho=2,
                        U=0,
                        V=0,
                        p=3,
                    ),
                ),
            ),
            right=RiemannBCState(
                alpha=0.9,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1950,
                        U=0,
                        V=0,
                        p=1000,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=1,
                    ),
                ),
            ),
            scheme=ToroScheme(
                eos=TwoPhaseEOS(
                    phase1=StiffenedGas(gamma=3.0, p0=3400),
                    phase2=PerfectGas(gamma=1.35),
                ),
                closure=Classical(),
            ),
            discontinuity_x0=0.5,
            final_time=0.15,
            CFL=0.45,
        ),
        # Test # 5
        # -------
        # The same as the test #3 in :cite:`tokareva_toro`
        RiemannProblem(
            left=RiemannBCState(
                alpha=0.8,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=0.75,
                        V=0,
                        p=1,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=0.75,
                        V=0,
                        p=1,
                    ),
                ),
            ),
            right=RiemannBCState(
                alpha=0.3,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=0.125,
                        U=0,
                        V=0,
                        p=0.1,
                    ),
                    phase2=RiemannState(
                        rho=0.125,
                        U=0,
                        V=0,
                        p=0.1,
                    ),
                ),
            ),
            scheme=ToroScheme(
                eos=TwoPhaseEOS(
                    phase1=PerfectGas(gamma=1.4), phase2=PerfectGas(gamma=1.4)
                ),
                closure=Classical(),
            ),
            discontinuity_x0=0.5,
            final_time=0.15,
            CFL=0.45,
        ),
        # Test # 6
        # -------
        # The same as the test #4 in :cite:`tokareva_toro`
        RiemannProblem(
            left=RiemannBCState(
                alpha=0.8,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=-2,
                        V=0,
                        p=0.4,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=-2,
                        V=0,
                        p=0.4,
                    ),
                ),
            ),
            right=RiemannBCState(
                alpha=0.5,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=2,
                        V=0,
                        p=0.4,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=2,
                        V=0,
                        p=0.4,
                    ),
                ),
            ),
            scheme=ToroScheme(
                eos=TwoPhaseEOS(
                    phase1=PerfectGas(gamma=1.4), phase2=PerfectGas(gamma=1.4)
                ),
                closure=Classical(),
            ),
            discontinuity_x0=0.5,
            final_time=0.15,
            CFL=0.45,
        ),
        # Test # 7
        # -------
        # The same as the test #5 in :cite:`tokareva_toro`
        RiemannProblem(
            left=RiemannBCState(
                alpha=0.6,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1.4,
                        U=0,
                        V=0,
                        p=2.0,
                    ),
                    phase2=RiemannState(
                        rho=1.4,
                        U=0,
                        V=0,
                        p=1.0,
                    ),
                ),
            ),
            right=RiemannBCState(
                alpha=0.3,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=3.0,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=0,
                        V=0,
                        p=1.0,
                    ),
                ),
            ),
            scheme=ToroScheme(
                eos=TwoPhaseEOS(
                    phase1=StiffenedGas(gamma=3.0, p0=10),
                    phase2=PerfectGas(gamma=1.4),
                ),
                closure=Classical(),
            ),
            discontinuity_x0=0.5,
            final_time=0.15,
            CFL=0.45,
        ),
        # Test # 8
        # -------
        # The same as the test #6 in :cite:`tokareva_toro`
        RiemannProblem(
            left=RiemannBCState(
                alpha=0.7,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=-19.5975,
                        V=0,
                        p=1000,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=-19.5975,
                        V=0,
                        p=1000,
                    ),
                ),
            ),
            right=RiemannBCState(
                alpha=0.2,
                state=PhasePair(
                    phase1=RiemannState(
                        rho=1,
                        U=-19.5975,
                        V=0,
                        p=0.01,
                    ),
                    phase2=RiemannState(
                        rho=1,
                        U=-19.5975,
                        V=0,
                        p=0.01,
                    ),
                ),
            ),
            scheme=ToroScheme(
                eos=TwoPhaseEOS(
                    phase1=StiffenedGas(gamma=3.0, p0=100),
                    phase2=PerfectGas(gamma=1.4),
                ),
                closure=Classical(),
            ),
            discontinuity_x0=0.8,
            final_time=0.007,
            CFL=0.45,
        ),
    ]

    yield tests


def set_bc_state(bc_state: RiemannBCState, eos: TwoPhaseEOS):
    fields = Q.fields
    state_array: Q = np.zeros(len(fields)).view(Q)

    alpha = bc_state.alpha
    state_array[..., fields.alpha] = alpha

    alphas = PhasePair(alpha, 1 - alpha)

    for phase, phase_data in bc_state.state.items():
        alpha = alphas[phase]
        phase_eos = eos[phase]
        rho = phase_data.rho
        U = phase_data.U
        rhoU = rho * U
        V = phase_data.V
        rhoV = rho * V
        p = phase_data.p

        rhoe = phase_eos.rhoe(rho, p)
        E = rhoe / rho + 0.5 * (U**2 + V**2)
        rhoE = rho * E
        c = phase_eos.sound_velocity(rho, p)

        state_array.set_phase(
            phase,
            np.asarray(
                (
                    alpha * rho,
                    alpha * rhoU,
                    alpha * rhoV,
                    alpha * rhoE,
                    rhoe,
                    U,
                    V,
                    p,
                    c,
                )
            ),
        )

    return state_array


def plot_func(data, time_annotation, lines, axes, fields_to_plot):
    t = data[0]
    time_annotation.set_text(f"t={t: .3f}s")
    x = data[1]
    values = data[2]

    # Alpha
    alpha = values[..., Q.fields.alpha]
    line = lines[0]
    line.set_data(x, alpha)
    ax = axes[0]
    ax.relim()
    ax.autoscale_view()

    for i, field in enumerate(fields_to_plot, 1):
        line = lines[i]
        line.set_data(x, values[..., field])
        ax = axes[i]
        ax.relim()
        ax.autoscale_view()


def test_toro(riemann_states, request, plot, write):
    for riemann in riemann_states:
        left = Line([0, 0], [0, 1])
        bottom = Line([0, 0], [1, 0])
        right = Line([1, 0], [1, 1])
        top = Line([0, 1], [1, 1])

        # BC
        Q_left: Q = set_bc_state(riemann.left, riemann.scheme.problem.eos)
        Q_right: Q = set_bc_state(riemann.right, riemann.scheme.problem.eos)

        left.bc = Dirichlet(Q_left)
        right.bc = Dirichlet(Q_right)
        top.bc = None
        bottom.bc = None

        mesh = Mesh(left, bottom, right, top, SimpleCell)
        mesh.interpolate(100, 1)
        mesh.generate()

        def init_fun(cells: MeshCellSet):
            xc = cells.centroids[..., 0]

            cells.values[np.where(xc > riemann.discontinuity_x0), ...] = Q_right
            cells.values[np.where(xc <= riemann.discontinuity_x0), ...] = Q_left

        scheme = riemann.scheme
        solver = BaerSolver(mesh, scheme)
        solver.init(init_fun)

        final_time = riemann.final_time
        t = 0
        CFL = riemann.CFL

        # :: Plot stuff ::
        fields = Q.fields
        fields_to_plot = [
            fields.arho1,
            fields.arho2,
            fields.U1,
            fields.U2,
            fields.p1,
            fields.p2,
        ]
        num_fields = len(fields_to_plot) // 2

        time_series = []
        artists = []
        axes = []

        x = solver.mesh.cells.centroids[..., 0]
        x = x.reshape(x.size)

        fig = plt.figure()

        # First plot alpha
        num_fields += 1
        gs = GridSpec(num_fields, 2)
        ax: plt.Axes = fig.add_subplot(gs[0, :])
        alpha = solver.mesh.cells.values[..., fields.alpha].ravel()
        (line,) = ax.plot(x, alpha, label=r"$\alpha$")
        ax.legend(loc="best")
        time_annotation = fig.text(0.5, 0.05, "t=0.00s", horizontalalignment="center")
        artists.append(line)
        axes.append(ax)

        for i, field in enumerate(fields_to_plot, 2):
            # Indices in the plot grid
            idx, idy = np.unravel_index(i, (num_fields, 2))
            ax: plt.Axes = fig.add_subplot(gs[idx, idy])
            field_value = solver.mesh.cells.values[..., field].ravel()
            (line,) = ax.plot(x, field_value, label=field.name)
            ax.legend(loc="best")
            artists.append(line)
            axes.append(ax)

        # :: End Plot Stuff ::

        while t <= final_time:
            # :: Plot Stuff ::
            x = solver.mesh.cells.centroids[..., 0]
            x = x.reshape(x.size)

            time_series.append(
                (
                    t,
                    x,
                    np.copy(solver.mesh.cells.values).view(Q),
                )
            )

            # :: End Plot Stuff ::

            dt = scheme.CFL(
                solver.mesh.cells,
                CFL,
            )
            assert ~np.isnan(dt)
            solver.step(dt)

            t += dt
            print(f"Time: {t}, dt: {dt}")

        fig.tight_layout()
        fig.subplots_adjust(
            bottom=0.15,
            top=0.95,
            hspace=0.35,
        )
        ani = FuncAnimation(
            fig,
            plot_func,
            [(data[0], data[1], data[2]) for data in time_series],
            fargs=(time_annotation, artists, axes, fields_to_plot),
            repeat=False,
        )

        if write:
            ani.save(f"twophase-{request.node.name}.mp4")

        if plot:
            plt.show()

        plt.close()
