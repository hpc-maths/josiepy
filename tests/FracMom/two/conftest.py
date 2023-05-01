import pytest

from josie.mesh.cellset import MeshCellSet

from josie.frac_mom.fields import FracMomFields

import numpy as np


def delta_shock(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]
    U = cells.values[..., FracMomFields.U].copy()
    V = cells.values[..., FracMomFields.V].copy()

    tol = 1e-13

    cells.values[..., FracMomFields.m0] = 1.0 / 10
    cells.values[..., FracMomFields.m12] = 1.0 / 10
    cells.values[..., FracMomFields.m1] = 1.0 / 10
    cells.values[..., FracMomFields.m32] = 1.0 / 10
    U = np.where(c_x > 0 + tol, -0.25, U)
    U = np.where(c_x < 0 + tol, 0.25, U)
    V = np.where(c_y > 0 + tol, -0.25, V)
    V = np.where(c_y < 0 + tol, 0.25, V)

    cells.values[..., FracMomFields.U] = U
    cells.values[..., FracMomFields.V] = V

    cells.values[..., FracMomFields.m1U] = U * cells.values[..., FracMomFields.m1]
    cells.values[..., FracMomFields.m1V] = V * cells.values[..., FracMomFields.m1]


def bouchut_vacuum(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]

    tol = 1e-13

    cells.values[..., FracMomFields.m0] = 0.5
    cells.values[..., FracMomFields.m12] = 0.5
    cells.values[..., FracMomFields.m1] = 0.5
    cells.values[..., FracMomFields.m32] = 0.5
    U = cells.values[..., FracMomFields.U].copy()
    V = cells.values[..., FracMomFields.V].copy()

    U = np.where(c_x > 0 + tol, 0.4, -0.4)
    V = np.where(c_y > 0 + tol, 0.4, -0.4)

    cells.values[..., FracMomFields.U] = U
    cells.values[..., FracMomFields.V] = V

    cells.values[..., FracMomFields.m1U] = U * cells.values[..., FracMomFields.m1]
    cells.values[..., FracMomFields.m1V] = V * cells.values[..., FracMomFields.m1]


def delta_shock2(cells: MeshCellSet):
    tol = 1e-13
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]
    c1_x = -0.15
    c1_y = -0.15
    c2_x = 0.15
    c2_y = 0.15
    sigma = 125.0 * 1000.0 / (33.0**2)
    g = np.exp(-sigma * (np.fabs(c_x - c1_x) ** 2 + np.fabs(c_y - c1_y) ** 2)) + np.exp(
        -sigma * (np.fabs(c_x - c2_x) ** 2 + np.fabs(c_y - c2_y) ** 2)
    )
    cells.values[..., FracMomFields.m0] = g
    cells.values[..., FracMomFields.m12] = g
    cells.values[..., FracMomFields.m1] = g
    cells.values[..., FracMomFields.m32] = g

    U = cells.values[..., FracMomFields.U].copy()
    V = cells.values[..., FracMomFields.V].copy()

    U = np.where(c_y > -c_x + tol, -1.0, 1.0)
    V = np.where(c_y > -c_x + tol, -1.0, 1.0)

    cells.values[..., FracMomFields.U] = U
    cells.values[..., FracMomFields.V] = V

    cells.values[..., FracMomFields.m1U] = (
        cells.values[..., FracMomFields.U] * cells.values[..., FracMomFields.m1]
    )
    cells.values[..., FracMomFields.m1V] = (
        cells.values[..., FracMomFields.V] * cells.values[..., FracMomFields.m1]
    )


def step(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]

    cells.values[..., FracMomFields.m0] = np.where(
        (c_x > 0.4) * (c_y > 0.4) * (c_x < 0.6) * (c_y < 0.6), 1, 1e-10
    )
    cells.values[..., FracMomFields.m12] = cells.values[..., FracMomFields.m0].copy()
    cells.values[..., FracMomFields.m1] = cells.values[..., FracMomFields.m0].copy()
    cells.values[..., FracMomFields.m32] = cells.values[..., FracMomFields.m0].copy()

    cells.values[..., FracMomFields.U] = 1.0
    cells.values[..., FracMomFields.V] = 1.0

    cells.values[..., FracMomFields.m1U] = (
        cells.values[..., FracMomFields.U] * cells.values[..., FracMomFields.m1]
    )
    cells.values[..., FracMomFields.m1V] = (
        cells.values[..., FracMomFields.V] * cells.values[..., FracMomFields.m1]
    )


def gauss(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]
    c1_x = -0.15
    c1_y = -0.15
    sigma = 125.0 * 1000.0 / (33.0**2)
    g = np.exp(-sigma * (np.abs(c_x - c1_x) ** 2 + np.abs(c_y - c1_y) ** 2))

    cells.values[..., FracMomFields.m0] = g
    cells.values[..., FracMomFields.m12] = g
    cells.values[..., FracMomFields.m1] = g
    cells.values[..., FracMomFields.m32] = g

    cells.values[..., FracMomFields.U] = 1.0
    cells.values[..., FracMomFields.V] = 1.0

    cells.values[..., FracMomFields.m1U] = (
        cells.values[..., FracMomFields.m1] * cells.values[..., FracMomFields.U]
    )
    cells.values[..., FracMomFields.m1V] = (
        cells.values[..., FracMomFields.m1] * cells.values[..., FracMomFields.V]
    )


def const(cells: MeshCellSet):
    cells.values[..., FracMomFields.m0] = 1.0
    cells.values[..., FracMomFields.m12] = 1.0
    cells.values[..., FracMomFields.m1] = 1.0
    cells.values[..., FracMomFields.m32] = 1.0

    cells.values[..., FracMomFields.U] = 1.0
    cells.values[..., FracMomFields.V] = 1.0

    cells.values[..., FracMomFields.m1U] = (
        cells.values[..., FracMomFields.m1] * cells.values[..., FracMomFields.U]
    )
    cells.values[..., FracMomFields.m1V] = (
        cells.values[..., FracMomFields.m1] * cells.values[..., FracMomFields.V]
    )


init_funs = [delta_shock, delta_shock2, bouchut_vacuum, gauss, step, const]


@pytest.fixture(params=sorted(init_funs, key=id))
def init_fun(request):
    yield request.param
