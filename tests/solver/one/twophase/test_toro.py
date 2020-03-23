import matplotlib.pyplot as plt
import numpy as np
import pytest

from collections import namedtuple
from dataclasses import dataclass
from matplotlib.animation import ArtistAnimation

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
from josie.solver.twophase.state import PhasePair, Q
from josie.solver.twophase.solver import TwoPhaseSolver


class ToroScheme(ExplicitEuler, Rusanov):
    pass


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
                    rho=0, U=0, V=0, p=0, eos=PerfectGas(gamma=1.4)
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
                    rho=0, U=0, V=0, p=0, eos=PerfectGas(gamma=1.4)
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
    __import__("ipdb").set_trace()
    solver = TwoPhaseSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = 0.25
    t = 0
    CFL = riemann_problem.CFL

    if plot:
        ims = []
        fig = plt.figure()
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

    while t <= final_time:
        x = solver.mesh.centroids[..., 0]
        x = x.reshape(x.size)

        alpha = solver.values[..., Q.fields.alpha]
        alpha = alpha.reshape(alpha.size)

        rho = solver.values[..., Q.fields.rho1]
        rho = rho.reshape(rho.size)

        U = solver.values[..., Q.fields.U1]
        U = U.reshape(U.size)

        p = solver.values[:, :, Q.fields.p1]
        p = p.reshape(p.size)

        if plot:
            (im1,) = ax1.plot(x, rho, "k-")
            ax1.set_xlabel("x")
            ax1.set_ylabel(r"$\rho$")

            (im2,) = ax2.plot(x, U, "k-")
            ax2.set_xlabel("x")
            ax2.set_ylabel("U")

            (im3,) = ax3.plot(x, p, "k-")
            ax3.set_xlabel("x")
            ax3.set_ylabel("p")

            (im4,) = ax4.plot(x, alpha, "k-")
            ax4.set_xlabel("x")
            ax4.set_ylabel(r"$\alpha$")

            ims.append([im1, im2, im3, im4])

        dt = scheme.CFL(
            solver.values,
            solver.mesh.volumes,
            solver.mesh.normals,
            solver.mesh.surfaces,
            CFL,
        )
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        plt.tight_layout()
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
        plt.close()
