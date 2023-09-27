# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import pytest

from dataclasses import dataclass

import josie.general.schemes.time as time_schemes
from josie.euler.schemes import EulerScheme
from josie.general.schemes.space import Godunov

# Test Toro


@dataclass
class RiemannSolution:
    rho_star_L: float
    rho_star_R: float
    p_star: float
    U_star: float


@dataclass
class RiemannState:
    rho: float
    U: float
    V: float
    p: float


@dataclass
class RiemannProblem:
    left: RiemannState
    right: RiemannState
    final_time: float
    CFL: float
    solution: RiemannSolution


riemann_states = [
    RiemannProblem(
        left=RiemannState(rho=1.0, U=0, V=0, p=1.0),
        right=RiemannState(rho=0.125, U=0, V=0, p=0.1),
        final_time=0.25,
        CFL=0.5,
        solution=RiemannSolution(
            p_star=0.30313,
            U_star=0.92745,
            rho_star_L=0.42632,
            rho_star_R=0.26557,
        ),
    ),
    RiemannProblem(
        left=RiemannState(rho=1.0, U=-2, V=0, p=0.4),
        right=RiemannState(rho=1.0, U=2.0, V=0, p=0.4),
        final_time=0.15,
        CFL=0.5,
        solution=RiemannSolution(
            p_star=0.00189,
            U_star=0.0,
            rho_star_L=0.02185,
            rho_star_R=0.02185,
        ),
    ),
    RiemannProblem(
        left=RiemannState(rho=1.0, U=0, V=0, p=1000),
        right=RiemannState(rho=1.0, U=0, V=0, p=0.01),
        final_time=0.012,
        CFL=0.45,
        solution=RiemannSolution(
            p_star=460.894,
            U_star=19.5975,
            rho_star_L=0.57506,
            rho_star_R=5.99924,
        ),
    ),
    RiemannProblem(
        left=RiemannState(rho=1.0, U=0, V=0, p=0.01),
        right=RiemannState(rho=1.0, U=0, V=0, p=100),
        final_time=0.035,
        CFL=0.45,
        solution=RiemannSolution(
            p_star=46.0950,
            U_star=-6.19633,
            rho_star_L=5.9924,
            rho_star_R=0.57511,
        ),
    ),
    RiemannProblem(
        left=RiemannState(rho=5.99924, U=19.5975, V=0, p=460.894),
        right=RiemannState(rho=5.9924, U=-6.19633, V=0, p=46.0950),
        final_time=0.035,
        CFL=0.5,
        solution=RiemannSolution(
            p_star=1691.64,
            U_star=8.68975,
            rho_star_L=14.2823,
            rho_star_R=31.0426,
        ),
    ),
]


@pytest.fixture(params=sorted(riemann_states, key=id))
def toro_riemann_state(request):
    yield request.param


@pytest.fixture(
    params=sorted(
        [member[1] for member in inspect.getmembers(time_schemes, inspect.isclass)],
        key=lambda c: c.__name__,
    ),
)
def TimeScheme(request):
    yield request.param


@pytest.fixture(params=sorted(EulerScheme._all_subclasses(), key=lambda c: c.__name__))
def IntercellFluxScheme(request):
    yield request.param


@pytest.fixture
def Scheme(IntercellFluxScheme, TimeScheme):
    """Create all the different schemes"""

    class ToroScheme(TimeScheme, Godunov, IntercellFluxScheme):
        pass

    return ToroScheme
