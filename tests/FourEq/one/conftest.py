# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dataclasses import dataclass

from josie.FourEq.exact import Exact
from josie.general.schemes.time.euler import ExplicitEuler
from josie.general.schemes.space.muscl import MUSCL_Hancock
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
        left=RiemannState(alpha=1.0 - eps, rho1=100.0, rho2=1e4, U=0.0),
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

    class CVVScheme(MUSCL_Hancock, MinMod, IntercellFlux, ExplicitEuler):
        pass

    return CVVScheme
