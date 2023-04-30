import pytest

from josie.mesh.cellset import MeshCellSet

from josie.pgd.fields import PGDFields

import numpy as np


def bouchut_vacuum(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    cells.values[..., PGDFields.rho] = 0.5
    U = cells.values[..., PGDFields.U].copy()

    U = np.where(c_x >= 0.5, 0.4, -0.4)
    cells.values[..., PGDFields.rhoU] = cells.values[..., PGDFields.rho] * U
    cells.values[..., PGDFields.U] = U
    cells.values[..., PGDFields.rhoV] = 0
    cells.values[..., PGDFields.V] = 0


def delta_shock(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    cells.values[..., PGDFields.rho] = np.power(np.sin(2 * np.pi * c_x), 4)
    U = cells.values[..., PGDFields.U].copy()

    U = np.where(c_x >= 0.5, -1.0, 1.0)
    cells.values[..., PGDFields.rhoU] = cells.values[..., PGDFields.rho] * U
    cells.values[..., PGDFields.U] = U
    cells.values[..., PGDFields.rhoV] = 0
    cells.values[..., PGDFields.V] = 0


def delta_shock2(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    cells.values[..., PGDFields.rho] = np.power(np.sin(2 * np.pi * c_x), 4)
    U = cells.values[..., PGDFields.U].copy()

    U = np.where(c_x > 0.5, -c_x, 1.0 - c_x)
    cells.values[..., PGDFields.rhoU] = cells.values[..., PGDFields.rho] * U
    cells.values[..., PGDFields.U] = U
    cells.values[..., PGDFields.rhoV] = 0
    cells.values[..., PGDFields.V] = 0


def gauss(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    cells.values[..., PGDFields.rho] = np.where(
        (c_x > 0.25) * (c_x < 0.75),
        np.power(np.cos(np.pi * (2 * c_x - 1)), 4),
        1e-10,
    )
    cells.values[..., PGDFields.U] = 1
    cells.values[..., PGDFields.rhoU] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.U]
    )
    cells.values[..., PGDFields.rhoV] = 0.0
    cells.values[..., PGDFields.V] = 0.0


def step(cells: MeshCellSet):
    c_x = cells.centroids[..., 0]
    cells.values[..., PGDFields.rho] = np.where(c_x > 0.45, 1, 1e-10)
    cells.values[..., PGDFields.U] = 1
    cells.values[..., PGDFields.rhoU] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.U]
    )
    cells.values[..., PGDFields.rhoV] = 0.0
    cells.values[..., PGDFields.V] = 0.0


def const(cells: MeshCellSet):
    cells.values[..., PGDFields.rho] = 1.0
    cells.values[..., PGDFields.U] = 1.0
    cells.values[..., PGDFields.rhoU] = (
        cells.values[..., PGDFields.rho] * cells.values[..., PGDFields.U]
    )
    cells.values[..., PGDFields.rhoV] = 0.0
    cells.values[..., PGDFields.V] = 0.0


init_funs = [delta_shock, delta_shock2, bouchut_vacuum, step, const]


@pytest.fixture(params=sorted(init_funs, key=id))
def init_fun(request):
    yield request.param
