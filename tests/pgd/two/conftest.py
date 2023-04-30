import pytest

from josie.mesh.cellset import MeshCellSet

from josie.pgd.fields import PGDFields

import numpy as np


def bouchut_vacuum(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]

    tol = 1e-13

    cells.values[..., PGDFields.rho] = 0.5
    U = cells.values[..., PGDFields.U].copy()
    V = cells.values[..., PGDFields.V].copy()
    U = np.where(c_x > 0 + tol, 0.4, -0.4)
    V = np.where(c_y > 0 + tol, 0.4, -0.4)

    cells.values[..., PGDFields.rhoU] = cells.values[..., PGDFields.rho] * U
    cells.values[..., PGDFields.U] = U
    cells.values[..., PGDFields.rhoV] = cells.values[..., PGDFields.rho] * V
    cells.values[..., PGDFields.V] = V


def delta_shock(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]

    tol = 1e-13

    cells.values[..., PGDFields.rho] = 1.0 / 10
    U = cells.values[..., PGDFields.U].copy()
    V = cells.values[..., PGDFields.V].copy()
    U = np.where(c_x > 0 + tol, -0.25, U)
    U = np.where(c_x < 0 + tol, 0.25, U)
    V = np.where(c_y > 0 + tol, -0.25, V)
    V = np.where(c_y < 0 + tol, 0.25, V)

    cells.values[..., PGDFields.rhoU] = cells.values[..., PGDFields.rho] * U
    cells.values[..., PGDFields.U] = U
    cells.values[..., PGDFields.rhoV] = cells.values[..., PGDFields.rho] * V
    cells.values[..., PGDFields.V] = V


def step(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]

    cells.values[..., PGDFields.rho] = np.where(
        (c_x > 0.4) * (c_y > 0.4) * (c_x < 0.6) * (c_y < 0.6), 1, 1e-10
    )

    cells.values[..., PGDFields.U] = 1.0
    cells.values[..., PGDFields.V] = 1.0

    cells.values[..., PGDFields.rhoU] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.U]
    )
    cells.values[..., PGDFields.rhoV] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.V]
    )


def gauss(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    c_y = cells.centroids[..., 1]

    xc = 0.0
    yc = 0.0
    sigma = 125.0 * 1000.0 / (33.0**2)

    cells.values[..., PGDFields.rho] = np.exp(
        -sigma * (np.fabs(c_x - xc) ** 2 + np.fabs(c_y - yc) ** 2)
    )

    cells.values[..., PGDFields.U] = 1.0
    cells.values[..., PGDFields.V] = 1.0

    cells.values[..., PGDFields.rhoU] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.U]
    )
    cells.values[..., PGDFields.rhoV] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.V]
    )


def const(cells: MeshCellSet):
    cells.values[..., PGDFields.rho] = 1.0

    cells.values[..., PGDFields.U] = 1.0
    cells.values[..., PGDFields.V] = 1.0

    cells.values[..., PGDFields.rhoU] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.U]
    )
    cells.values[..., PGDFields.rhoV] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.V]
    )


init_funs = [delta_shock, bouchut_vacuum, gauss, step, const]


@pytest.fixture(params=sorted(init_funs, key=id))
def init_fun(request):
    yield request.param
