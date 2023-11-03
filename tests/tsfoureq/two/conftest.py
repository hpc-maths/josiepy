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
    small_scale: RiemannState
    final_time: float
    xd: float
    CFL: float


eps1 = 0.1
eps2 = 1e-7

riemann_states = [
    # RiemannProblem(
    #     left=RiemannState(
    #         alphabar=1.0 - eps1, ad=0, rho1=1.0, rho2=1.0e3, U=0.15
    #     ),
    #     right=RiemannState(
    #         alphabar=eps1, ad=0, rho1=1.0, rho2=1.0e3, U=0.15
    #     ),
    #     final_time=3.33,
    #     xd=0.25,
    #     CFL=0.5,
    # ),
    RiemannProblem(
        left=RiemannState(alphabar=1.0 - eps2, ad=0.0, rho1=100.0, rho2=1e4, U=0.0),
        right=RiemannState(alphabar=eps2, ad=0, rho1=1.0, rho2=1e3, U=0.0),
        small_scale=RiemannState(alphabar=eps2, ad=0.4, rho1=1.0, rho2=1e3, U=0.0),
        final_time=9e-2,
        xd=0.3,
        CFL=0.5,
    ),
    # RiemannProblem(
    #     left=RiemannState(
    #         alphabar=1.0 - eps2, ad=0, rho1=100.0, rho2=1e4, U=0.0
    #     ),
    #     right=RiemannState(
    #         alphabar=eps2, ad=0, rho1=1.0, rho2=1e3, U=0.0
    #     ),
    #     small_scale=RiemannState(
    #         alphabar=eps2, ad=0.2, rho1=1.0, rho2=1e3, U=0.0
    #     ),
    #     final_time=0.03,
    #     xd=0.65,
    #     CFL=0.5,
    # ),
    # RiemannProblem(
    #     left=RiemannState(
    #         alphabar=1.0 - eps2, ad=0.2, rho1=100.0, rho2=1e4, U=0.0
    #     ),
    #     right=RiemannState(
    #         alphabar=eps2, ad=0.2, rho1=1.0, rho2=1e3, U=0.0
    #     ),
    #     final_time=0.03,
    #     xd=0.3,
    #     CFL=0.5,
    # ),
    # RiemannProblem(
    #     left=RiemannState(
    #         alphabar=1.0 - eps2, ad=0, rho1=100.0, rho2=1e4, U=0.0
    #     ),
    #     right=RiemannState(
    #         alphabar=eps2, ad=0.2, rho1=1.0, rho2=1e3, U=0.0
    #     ),
    #     final_time=0.03,
    #     xd=0.3,
    #     CFL=0.5,
    # ),
    # RiemannProblem(
    #     left=RiemannState(
    #         alphabar=1.0 - eps2, ad=0.2, rho1=100.0, rho2=1e4, U=0.0
    #     ),
    #     right=RiemannState(
    #         alphabar=eps2, ad=0, rho1=1.0, rho2=1e3, U=0.0
    #     ),
    #     final_time=0.03,
    #     xd=0.3,
    #     CFL=0.5,
    # ),
]


@pytest.fixture(params=sorted(riemann_states, key=id))
def riemann_state(request):
    yield request.param
