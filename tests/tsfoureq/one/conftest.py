# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dataclasses import dataclass


@dataclass
class RiemannState:
    alphabar: float
    ad: float
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


@dataclass
class RiemannExactProblem:
    left: RiemannState
    right: RiemannState
    final_time: float
    xd: float
    CFL: float
    left_star: RiemannState
    right_star: RiemannState


eps2 = 1e-7

riemann_states = [
    RiemannExactProblem(
        left=RiemannState(alphabar=1.0 - eps2, ad=0.2, rho1=100.0, rho2=1e4, U=0.0),
        right=RiemannState(alphabar=eps2, ad=0.2, rho1=1.0, rho2=1e3, U=0.0),
        final_time=0.03,
        xd=0.3,
        CFL=0.5,
        left_star=RiemannState(
            alphabar=1.0 - eps2,
            ad=0.19691374708,
            rho1=98.0785047004,
            rho2=9807.85047004,
            U=0.0581402338983,
        ),
        right_star=RiemannState(
            alphabar=eps2,
            ad=0.200620960991,
            rho1=1.00388402097,
            rho2=1003.88402097,
            U=0.0581402338983,
        ),
    ),
    RiemannExactProblem(
        left=RiemannState(alphabar=1.0 - eps2, ad=0, rho1=100.0, rho2=1e4, U=0.0),
        right=RiemannState(alphabar=eps2, ad=0.2, rho1=1.0, rho2=1e3, U=0.0),
        final_time=0.03,
        xd=0.3,
        CFL=0.5,
        left_star=RiemannState(
            alphabar=1.0 - eps2,
            ad=0,
            rho1=98.0808333954,
            rho2=9808.08333954,
            U=0.0581416258859,
        ),
        right_star=RiemannState(
            alphabar=eps2,
            ad=0.2,
            rho1=1.00388411414,
            rho2=1003.88411414,
            U=0.0581416258859,
        ),
    ),
    RiemannExactProblem(
        left=RiemannState(alphabar=1.0 - eps2, ad=0.2, rho1=100.0, rho2=1e4, U=0.0),
        right=RiemannState(alphabar=eps2, ad=0, rho1=1.0, rho2=1e3, U=0.0),
        final_time=0.03,
        xd=0.3,
        CFL=0.5,
        left_star=RiemannState(
            alphabar=1.0 - eps2,
            ad=0.2,
            rho1=98.0782714158,
            rho2=9807.82714158,
            U=0.0581473615097,
        ),
        right_star=RiemannState(
            alphabar=eps2,
            ad=0,
            rho1=1.00388401164,
            rho2=1003.88401164,
            U=0.0581473615097,
        ),
    ),
]


@pytest.fixture(params=sorted(riemann_states, key=id))
def riemann_state(request):
    yield request.param
