import pytest

from josie.mesh.cellset import MeshCellSet

from josie.frac_mom.fields import FracMomFields

import numpy as np


def delta_shock(cells: MeshCellSet):
    xc = cells.centroids[..., 0]
    c1 = 0.15
    c2 = 0.85
    sigmax = 0.075
    Smin = 0.3
    Smax = 0.7

    def g(k):
        return (2.0 / k) * (np.power(Smax, 0.5 * k) - np.power(Smin, 0.5 * k)) * np.exp(
            -((xc - c1) ** 2) / (sigmax**2)
        ) + (2.0 / k) * (np.power(Smax, 0.5 * k) - np.power(Smin, 0.5 * k)) * np.exp(
            -((xc - c2) ** 2) / (sigmax**2)
        )

    cells.values[..., FracMomFields.m0] = g(2)
    cells.values[..., FracMomFields.m12] = g(3)
    cells.values[..., FracMomFields.m1] = g(4)
    cells.values[..., FracMomFields.m32] = g(5)

    U = cells.values[..., FracMomFields.U].copy()
    U = np.where(xc > 0.5, -0.5, 0.5)

    cells.values[..., FracMomFields.m1U] = cells.values[..., FracMomFields.m1] * U
    cells.values[..., FracMomFields.U] = U
    cells.values[..., FracMomFields.m1V] = 0.0
    cells.values[..., FracMomFields.V] = 0.0


def bouchut_vacuum(cells: MeshCellSet):
    xc = cells.centroids[..., 0]
    cells.values[..., FracMomFields.m0] = 0.5
    cells.values[..., FracMomFields.m12] = 0.5
    cells.values[..., FracMomFields.m1] = 0.5
    cells.values[..., FracMomFields.m32] = 0.5

    U = cells.values[..., FracMomFields.U].copy()
    U = np.where(xc < 0.5, -0.4, 0.4)

    cells.values[..., FracMomFields.m1U] = cells.values[..., FracMomFields.m1] * U
    cells.values[..., FracMomFields.U] = U
    cells.values[..., FracMomFields.m1V] = 0.0
    cells.values[..., FracMomFields.V] = 0.0


def delta_shock2(cells: MeshCellSet):
    xc = cells.centroids[..., 0]
    cells.values[..., FracMomFields.m0] = 0.5
    cells.values[..., FracMomFields.m12] = 0.5
    cells.values[..., FracMomFields.m1] = 0.5
    cells.values[..., FracMomFields.m32] = 0.5

    U = cells.values[..., FracMomFields.U].copy()
    U = np.where(xc >= 0.5, -xc, -xc + 1.0)

    cells.values[..., FracMomFields.m1U] = cells.values[..., FracMomFields.m1] * U
    cells.values[..., FracMomFields.U] = U
    cells.values[..., FracMomFields.m1V] = 0.0
    cells.values[..., FracMomFields.V] = 0.0


def step(cells: MeshCellSet):
    xc = cells.centroids[..., 0]
    cells.values[..., FracMomFields.m0] = np.where(xc > 0.45, 1, 1e-10)
    cells.values[..., FracMomFields.m12] = np.where(xc > 0.45, 1, 1e-10)
    cells.values[..., FracMomFields.m1] = np.where(xc > 0.45, 1, 1e-10)
    cells.values[..., FracMomFields.m32] = np.where(xc > 0.45, 1, 1e-10)

    cells.values[..., FracMomFields.U] = 1
    cells.values[..., FracMomFields.m1U] = (
        cells.values[..., FracMomFields.U] * cells.values[..., FracMomFields.m1]
    )
    cells.values[..., FracMomFields.m1V] = 0.0
    cells.values[..., FracMomFields.V] = 0.0


def const(cells: MeshCellSet):
    cells.values[..., FracMomFields.m0] = 1.0
    cells.values[..., FracMomFields.m12] = 1.0
    cells.values[..., FracMomFields.m1] = 1.0
    cells.values[..., FracMomFields.m32] = 1.0

    cells.values[..., FracMomFields.U] = 1
    cells.values[..., FracMomFields.m1U] = (
        cells.values[..., FracMomFields.U] * cells.values[..., FracMomFields.m1]
    )
    cells.values[..., FracMomFields.m1V] = 0.0
    cells.values[..., FracMomFields.V] = 0.0


init_funs = [delta_shock, delta_shock2, bouchut_vacuum, step, const]


@pytest.fixture(params=sorted(init_funs, key=id))
def init_fun(request):
    yield request.param
