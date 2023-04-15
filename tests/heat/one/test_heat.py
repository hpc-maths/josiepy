# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

r""" Tests for the heat equation

.. math::
    \frac{\partial T}{\partial t} = \frac{\partial^2 T}{\partial x^2} + b(x, t)

with:

* :math:`b(x, t) = (4 \pi^2 - 1)\sin(2 \pi x)e^{-t}`
* Solution :math:`u(x,t) = \sin(2 \pi x) e^{-t}`

https://gitlab.com/rubendibattista/josiepy/-/snippets/2082802

"""
import numpy as np
import pytest

import matplotlib.pyplot as plt

from josie.math import Direction
from josie.general.schemes.time import ExplicitEuler
from josie.boundary.boundary import Line
from josie.bc import Dirichlet
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.heat.schemes import HeatScheme
from josie.heat.problem import HeatProblem
from josie.heat.solver import HeatSolver
from josie.heat.state import Q
from josie.heat.transport import ConstantHeatTransport
from josie.general.schemes.diffusive import CentralDifferenceGradient
from josie.general.schemes.source import ConstantSource


def relative_error(a, b):
    return np.abs(a - b)


def exact_solution(x, t):
    return np.sin(2 * np.pi * x) * np.exp(-t)


def b(x, t):
    return (4 * np.pi**2 - 1) * exact_solution(x, t)


class Problem(HeatProblem):
    def s(self, cells, t):
        dimensionality = cells.dimensionality
        x = cells.centroids[..., :dimensionality]
        # Negative because we consider everything on the left of the equal sign
        return -b(x, t).reshape(cells.values.shape)


def initial_condition(x):
    return exact_solution(x, 0)


def init_fun(cells):
    cells.values = initial_condition(cells.centroids[..., 0])[..., np.newaxis]


class CentralHeatScheme(
    HeatScheme, CentralDifferenceGradient, ConstantSource, ExplicitEuler
):
    pass


@pytest.fixture
def N():
    yield 500


@pytest.fixture
def boundaries(N):
    """1D problem along x"""

    # This is needed to have the same centroids coordinates as  for 1D
    # diffusion code here:
    L = -1 - 1 / (N - 1)
    R = 1 + 1 / (N - 1)
    left = Line([L, 0], [L, 1])
    bottom = Line([L, 0], [R, 0])
    right = Line([R, 0], [R, 1])
    top = Line([L, 1], [R, 1])

    left.bc = Dirichlet(Q(0))
    right.bc = Dirichlet(Q(0))
    top.bc = None
    bottom.bc = None

    yield (left, bottom, right, top)


@pytest.fixture()
def mesh(boundaries, N):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(N, 1)
    mesh.generate()

    yield mesh


def test_heat(mesh, plot):
    thermal_diffusivity = 1
    problem = Problem(ConstantHeatTransport(thermal_diffusivity))
    scheme = CentralHeatScheme(problem)
    solver = HeatSolver(mesh, scheme)
    solver.init(init_fun)

    CFL = 1
    t = 0.0
    final_time = 0.1

    cells = solver.mesh.cells

    while t <= final_time:
        dt = scheme.CFL(cells, CFL)
        solver.step(dt)

        t += dt
        print(f"Time: {solver.t}, dt: {dt}")

    # Final solution
    x = cells.centroids[..., Direction.X]
    x = x.reshape(x.size)

    T_exact = exact_solution(x, t)
    T = cells.values[..., Q.fields.T]

    assert t >= final_time

    tolerance = 1e-2
    err = relative_error(T.reshape(T_exact.size), T_exact)

    inliers = np.count_nonzero(err < tolerance) / err.size

    # 90% of the points are within tolerance
    assert inliers >= 0.9

    if plot:
        # Plot final step solution

        plt.plot(x, T, "--o", label=f"Numerical {t}")
        plt.plot(x, T_exact, label=f"Exact {t}")

        plt.tight_layout()

        plt.show()
        plt.close()
