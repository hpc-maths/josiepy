import numpy as np
import pytest

from josie.bc import Neumann
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import DGCell

from josie.general.schemes.time.rk import RK2
from josie.pgd.schemes.LF import LF
from josie.pgd.state import Q
from josie.pgd.solver import PGDSolver
from josie.pgd.fields import PGDFields
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
    mesh.interpolate(50, 1)
    mesh.generate()

    solver = PGDSolver(mesh, scheme)

    solver.init(init_fun)
    solver.scheme.init_limiter(solver.mesh.cells)

    yield solver


def test_pgd(solver, plot):
    """Testing against the real 1D solver"""

    rLGLmin = 2.0
    cfl = 0.1
    dx = solver.mesh._x[1, 0] - solver.mesh._x[0, 0]
    tf = 0.5

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ims = []

    t = 0.0
    while t < tf:
        maxvel = np.amax(np.abs(solver.mesh.cells.values[..., PGDFields.U]))
        dt = cfl * rLGLmin * dx / maxvel
        x = solver.mesh.cells.centroids[..., 1, 0]
        rho = solver.mesh.cells.values[..., 1, PGDFields.rho]
        Ux = solver.mesh.cells.values[..., 1, PGDFields.U]
        if plot:
            (im1,) = ax1.plot(x, rho, "ro-")
            (im2,) = ax2.plot(x, Ux, "ro-")
            ims.append([im1, im2])

        solver.step(dt)
        t += dt

    if plot:
        _ = ArtistAnimation(fig, ims)
        # ani.save("PGD_1D_d-choc2_50x1.mp4", writer="ffmpeg")
        plt.show()
