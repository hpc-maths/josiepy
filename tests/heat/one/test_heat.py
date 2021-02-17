""" Tests for the heat equation

Source: https://web.stanford.edu/class/math220b/handouts/heateqn.pdf
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
from josie.transport import ConstantTransport
from josie.general.schemes.diffusive import CentralDifferenceGradient
from josie.general.schemes.source import ConstantSource


def exact_solution(x, t):
    return np.sin(2 * np.pi * x) * np.exp(-t)


def b(x, t):
    return (4 * np.pi ** 2 - 1) * exact_solution(x, t)


class Problem(HeatProblem):
    def s(self, cells, t):
        dimensionality = cells.dimensionality
        x = cells.centroids[..., :dimensionality]
        # Negative because we consider everything on the left of the equal sign
        return -b(x, t).reshape(cells.values.shape)


def initial_condition(x):
    return exact_solution(x, 0)


def init_fun(cells):
    cells.values = initial_condition(cells.centroids[..., 0])


class CentralHeatScheme(
    HeatScheme, CentralDifferenceGradient, ConstantSource, ExplicitEuler
):
    pass


@pytest.fixture
def N():
    yield 10


@pytest.fixture
def boundaries(N):
    """ 1D problem along x """
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
    problem = Problem(ConstantTransport(None, thermal_diffusivity))
    scheme = CentralHeatScheme(problem)
    solver = HeatSolver(mesh, scheme)
    solver.init(init_fun)

    CFL = 1
    t = 0.0
    final_time = 1

    cells = solver.mesh.cells

    while t <= final_time:
        dt = scheme.CFL(cells, CFL)
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        # Plot final step solution

        x = cells.centroids[..., Direction.X]
        x = x.reshape(x.size)

        T_exact = exact_solution(
            initial_condition, x, t, thermal_diffusivity, sum_elements=10
        )

        T = cells.values[..., Q.fields.T]
        plt.plot(x, T, "x")
        plt.plot(x, T_exact)

        plt.tight_layout()

        plt.show()
        plt.close()
