# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np

from josie.general.schemes.time.rk import RK2_relax, RK2
from josie.general.schemes.time.euler import ExplicitEuler
from josie.general.schemes.space.muscl import MUSCL
from josie.general.schemes.space.limiters import MinMod

from josie.ts_cap.schemes import Rusanov
from josie.ts_cap.exact import ExactHyp
from josie.ts_cap.arithmetic import ArithmeticCap
from josie.ts_cap.solver import TsCapSolver, TsCapLieSolver


def f(x):
    return np.maximum(np.exp(2 * x**2 * (x**2 - 3) / (x**2 - 1) ** 2), 0)


def g(x):
    k = 8
    return np.maximum(0.5 + 0.5 * np.tanh(-k * (x - 0.5)), 0)


def circle(
    R: float,
    x_c: np.ndarray,
    y_c: np.ndarray,
    x_0: float,
    y_0: float,
    symBool: bool = True,
):
    eps = R * 0.6
    r = np.sqrt((x_c - x_0) ** 2 + (y_c - y_0) ** 2)
    arr = np.where(
        (r >= R - eps / 2) * (r < R + eps / 2),
        g((r - R + eps / 2) / eps),
        np.where(r < R - eps / 2, 1, 0),
    )

    arr[arr < 1e-15] = 0

    if symBool:
        # Enforce symmetry along
        # X-axis
        arr = 0.5 * (arr + arr[::-1, :])
        # Y-axis
        arr = 0.5 * (arr + arr[:, ::-1])
        # XY-axis
        arr = 0.5 * (arr + np.transpose(arr, axes=(1, 0, 2)))

    return arr


def square(R: float, x_c: np.ndarray, y_c: np.ndarray, x_0: float, y_0: float):
    eps = R * 1.4

    # Square
    # Left/right sides
    arr = np.where(
        (np.abs(x_c - x_0) >= R - eps / 2) * (np.abs(x_c - x_0) < R + eps / 2),
        # Interface layer
        g((np.abs(x_c - x_0) - R + eps / 2) / eps),
        np.where(
            (np.abs(x_c - x_0) >= R + eps / 2),
            # Outside
            0,
            # Inside
            1,
        ),
    )
    arr *= np.where(
        (np.abs(y_c - y_0) >= R - eps / 2) * (np.abs(y_c - y_0) < R + eps / 2),
        # Interface layer
        g((np.abs(y_c - y_0) - R + eps / 2) / eps),
        np.where(
            (np.abs(y_c - y_0) >= R + eps / 2),
            # Outside
            0,
            # Inside
            1,
        ),
    )

    # Enforce symmetry along
    # X-axis
    arr = 0.5 * (arr + arr[::-1, :])
    # Y-axis
    arr = 0.5 * (arr + arr[:, ::-1])
    # XY-axis
    arr = 0.5 * (arr + np.transpose(arr, axes=(1, 0, 2)))

    return arr


@pytest.fixture(params=[0])
def nSmoothPass(request):
    yield request.param


@pytest.fixture(params=["HypCap-Splitting"])
def init_schemes(request):
    if request.param == "HypCap-SameFlux":

        class TsCapScheme(Rusanov, ExplicitEuler, MUSCL, MinMod):
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


@pytest.fixture(params=[square])
def shape_fun(request):
    yield request.param


@pytest.fixture(params=[1e3])
def Hmax(request):
    yield request.param


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
