# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np

from dataclasses import dataclass

from josie.twofluid.fields import Phases
from josie.FourEq.state import Q

from josie.FourEq.exact import Exact
from josie.FourEq.schemes import Rusanov
from josie.general.schemes.time.rk import RK2_relax
from josie.general.schemes.space.godunov import Godunov
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


@pytest.fixture(params=[1e-7, 0.0])
def epsilon(request):
    yield request.param


@pytest.fixture(params=["Advection", "Shock"])
def riemann_state(request, epsilon):
    if request.param == "Advection":
        yield RiemannProblem(
            left=RiemannState(alpha=1.0 - epsilon, rho1=1.0, rho2=1.0e3, U=0.15),
            right=RiemannState(alpha=epsilon, rho1=1.0, rho2=1.0e3, U=0.15),
            final_time=0.1,
            xd=0.25,
            CFL=0.8,
        )
    if request.param == "Shock":
        yield RiemannProblem(
            left=RiemannState(alpha=1.0 - epsilon, rho1=100.0, rho2=1e3 + 3.96, U=0.0),
            right=RiemannState(alpha=epsilon, rho1=1.0, rho2=1e3, U=0.0),
            final_time=0.03,
            xd=0.3,
            CFL=0.5,
        )


@pytest.fixture(params=[Exact, Rusanov])
def IntercellFlux(request):
    yield request.param


@pytest.fixture(params=["MUSCL", "Godunov"])
def Scheme(request, IntercellFlux):
    """Create all the different schemes"""

    if request.param == "Godunov":

        class GCVVScheme(IntercellFlux, RK2_relax, Godunov):
            pass

        return GCVVScheme

    if request.param == "MUSCL":

        class MCVVScheme(IntercellFlux, RK2_relax, MUSCL, MinMod):
            pass

        return MCVVScheme


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
        if state.alpha > 0.0:
            p1 = eos[Phases.PHASE1].p(state.rho1)
        else:
            p1 = np.nan
        if state.alpha < 1.0:
            p2 = eos[Phases.PHASE2].p(state.rho2)
        else:
            p2 = np.nan
        c1 = eos[Phases.PHASE1].sound_velocity(state.rho1)
        c2 = eos[Phases.PHASE2].sound_velocity(state.rho2)
        if state.alpha > 0.0:
            if state.alpha < 1.0:
                P = state.alpha * p1 + (1.0 - state.alpha) * p2
            else:
                P = p1
        else:
            P = p2
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
