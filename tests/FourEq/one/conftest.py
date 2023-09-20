# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np

from dataclasses import dataclass

from josie.twofluid.fields import Phases
from josie.FourEq.state import Q

from josie.FourEq.exact import Exact
from josie.general.schemes.time.rk import RK2
from josie.general.schemes.space.muscl import MUSCL
from josie.general.schemes.space.limiters import MinMod


@dataclass
class RiemannState:
    alpha: float
    rho1: float
    rho2: float
    U: float


@dataclass
class RiemannProblem:
    left: RiemannState
    right: RiemannState
    final_time: float
    xd: float
    CFL: float


eps = 1e-7

riemann_states = [
    RiemannProblem(
        left=RiemannState(alpha=1.0 - eps, rho1=1.0, rho2=1.0e3, U=0.15),
        right=RiemannState(alpha=eps, rho1=1.0, rho2=1.0e3, U=0.15),
        final_time=3.333,
        xd=0.25,
        CFL=0.5,
    ),
    RiemannProblem(
        left=RiemannState(alpha=1.0 - eps, rho1=100.0, rho2=1e3 + 3.96, U=0.0),
        right=RiemannState(alpha=eps, rho1=1.0, rho2=1e3, U=0.0),
        final_time=0.03,
        xd=0.3,
        CFL=0.5,
    ),
]


@pytest.fixture(params=sorted(riemann_states, key=id))
def riemann_state(request):
    yield request.param


@pytest.fixture(params=[Exact])
def IntercellFlux(request):
    yield request.param


@pytest.fixture
def Scheme(IntercellFlux):
    """Create all the different schemes"""

    class CVVScheme(IntercellFlux, RK2, MUSCL, MinMod):
        pass

    return CVVScheme


@pytest.fixture
def riemann2Q():
    def riemann2Q(state, eos):
        """Wrap all the operations to create a complete FourEq state from the
        initial Riemann Problem data
        """
        # BC
        arho1 = state.alpha * state.rho1
        arho2 = (1.0 - state.alpha) * state.rho2
        rho = arho1 + arho2
        arho = state.alpha * rho
        rhoU = rho * state.U
        rhoV = 0.0
        V = 0.0
        p1 = eos[Phases.PHASE1].p(state.rho1)
        p2 = eos[Phases.PHASE2].p(state.rho2)
        c1 = eos[Phases.PHASE1].sound_velocity(state.rho1)
        c2 = eos[Phases.PHASE2].sound_velocity(state.rho2)
        P = state.alpha * p1 + (1.0 - state.alpha) * p2
        c = np.sqrt((arho1 * c1**2 + arho2 * c2**2) / rho)

        return Q(
            arho,
            rhoU,
            rhoV,
            rho,
            state.U,
            V,
            P,
            c,
            state.alpha,
            arho1,
            p1,
            c1,
            arho2,
            p2,
            c2,
        )

    yield riemann2Q
