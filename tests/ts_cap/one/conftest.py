# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

from josie.general.schemes.time.rk import RK2_relax, RK2
from josie.general.schemes.space.muscl import MUSCL
from josie.general.schemes.space.limiters import MinMod

from josie.ts_cap.schemes import Rusanov
from josie.ts_cap.exact import ExactHyp
from josie.ts_cap.arithmetic import ArithmeticCap
from josie.ts_cap.solver import TsCapSolver, TsCapLieSolver


@pytest.fixture(params=["HypCap-SameFlux", "HypCap-Splitting"])
def init_schemes(request):
    if request.param == "HypCap-SameFlux":

        class TsCapScheme(Rusanov, RK2_relax, MUSCL, MinMod):
            pass

        def initschemes(eos, sigma, Hmax, kappa, dx, dy, norm_grada_min, nSmoothPass):
            return [
                TsCapScheme(
                    eos, sigma, Hmax, kappa, dx, dy, norm_grada_min, nSmoothPass
                )
            ]

        yield initschemes

    else:

        class TsCapHypScheme(ExactHyp, RK2_relax, MUSCL, MinMod):
            pass

        class TsCapCapScheme(ArithmeticCap, RK2, MUSCL, MinMod):
            pass

        def initschemes(eos, sigma, Hmax, kappa, dx, dy, norm_grada_min, nSmoothPass):
            return [
                TsCapHypScheme(
                    eos, sigma, Hmax, kappa, dx, dy, norm_grada_min, nSmoothPass
                ),
                TsCapCapScheme(
                    eos, sigma, Hmax, kappa, dx, dy, norm_grada_min, nSmoothPass
                ),
            ]

        yield initschemes


@pytest.fixture
def init_solver():
    """Init solver depending on a splitted choice of schemes or not"""

    def initsolver(mesh, schemes, initfun):
        if len(schemes) == 1:
            solver = TsCapSolver(mesh, schemes[0])
            schemes[0].tmp_arr = np.zeros_like(mesh.cells.centroids[..., 0])
            solver.init(initfun)
            solver.mesh.update_ghosts(0)
            solver.scheme.auxilliaryVariableUpdate(solver.mesh.cells._values)
            solver.mesh.update_ghosts(0)
        else:
            solver = TsCapLieSolver(mesh, schemes)
            schemes[0].tmp_arr = np.zeros_like(mesh.cells.centroids[..., 0])
            schemes[1].tmp_arr = np.zeros_like(mesh.cells.centroids[..., 0])
            solver.init(initfun)
            solver.mesh.update_ghosts(0)
            solver.schemes[0].auxilliaryVariableUpdate(solver.mesh.cells._values)
            solver.mesh.update_ghosts(0)

        return solver

    yield initsolver
