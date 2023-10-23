# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from josie.tsfoureq.exact import Exact
from josie.tsfoureq.schemes import Rusanov
from josie.general.schemes.time.rk import RK2_relax
from josie.general.schemes.space import Godunov
from josie.general.schemes.space.muscl import MUSCL
from josie.general.schemes.space.limiters import MinMod


@pytest.fixture(params=[Godunov, MUSCL])
def SpaceScheme(request):
    yield request.param


@pytest.fixture(params=[Rusanov, Exact])
def IntercellFlux(request):
    yield request.param


@pytest.fixture
def Scheme(SpaceScheme, IntercellFlux):
    """Create all the different schemes"""

    if SpaceScheme == MUSCL:

        class TSFourEqScheme(MUSCL, MinMod, IntercellFlux, RK2_relax):
            pass

    else:

        class TSFourEqScheme(SpaceScheme, IntercellFlux, RK2_relax):
            pass

    return TSFourEqScheme
