import numpy as np
import pytest

from josie.bc import Neumann
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import DGCell

from josie.general.schemes.time.rk import RK2
from josie.frac_mom.schemes.LF import LF
from josie.frac_mom.state import Q
from josie.frac_mom.solver import FracMomSolver
from josie.frac_mom.fields import FracMomFields
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


@pytest.fixture
def scheme():
    class Test_scheme(LF, RK2):
        pass

    yield Test_scheme()


@pytest.fixture
def solver(scheme, init_fun):
    """1D problem along x"""
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    dQ = np.zeros(len(Q.fields)).view(Q)
    left.bc = Neumann(dQ)
    right.bc = Neumann(dQ)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, DGCell)
    mesh.interpolate(100, 1)
    mesh.generate()

    solver = FracMomSolver(mesh, scheme)

    solver.init(init_fun)
    solver.scheme.init_limiter(solver.mesh.cells)
    solver.scheme.alpha = 1

    yield solver


def test_against_real_1D(solver, plot):
    """Testing against the real 1D solver"""

    rLGLmin = 2.0
    cfl = 0.1
    tf = 0.7

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ims = []

    while solver.t < tf:
        x = solver.mesh.cells.centroids[..., 1, 0]

        m0 = solver.mesh.cells.values[..., 1, FracMomFields.m0]
        # m12 = solver.mesh.cells.values[..., 1, FracMomFields.m12]
        # m1 = solver.mesh.cells.values[..., 1, FracMomFields.m1]
        # m32 = solver.mesh.cells.values[..., 1, FracMomFields.m32]
        Ux = solver.mesh.cells.values[..., 1, FracMomFields.U]

        if plot:
            (im1,) = ax1.plot(x, m0, "ro-")
            (im2,) = ax2.plot(x, Ux, "ro-")
            ims.append([im1, im2])

        dt = solver.scheme.CFL(solver.mesh.cells, cfl * rLGLmin)
        solver.step(dt)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
