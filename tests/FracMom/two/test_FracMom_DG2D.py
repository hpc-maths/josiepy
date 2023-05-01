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
    left = Line([-1, -1], [-1, 1])
    bottom = Line([-1, -1], [1, -1])
    right = Line([1, -1], [1, 1])
    top = Line([-1, 1], [1, 1])

    dQ = np.zeros(len(Q.fields)).view(Q)
    left.bc = Neumann(dQ)
    right.bc = Neumann(dQ)
    bottom.bc = Neumann(dQ)
    top.bc = Neumann(dQ)

    mesh = Mesh(left, bottom, right, top, DGCell)
    mesh.interpolate(50, 50)
    mesh.generate()

    solver = FracMomSolver(mesh, scheme)

    solver.init(init_fun)
    solver.scheme.init_limiter(solver.mesh.cells)

    yield solver


def test_against_real_1D(solver, plot):
    """Testing against the real 1D solver"""

    rLGLmin = 2.0
    cfl = 0.1
    tf = 0.4

    fig = plt.figure()
    ax3d = plt.axes(projection="3d")
    ims = []

    while solver.t < tf:
        dt = solver.scheme.CFL(solver.mesh.cells, cfl * rLGLmin)

        if plot:
            tabx = solver.mesh.cells.centroids[..., 1, 0]
            taby = solver.mesh.cells.centroids[..., 1, 1]
            tab_m0 = solver.mesh.cells.values[..., 1, FracMomFields.m0]
            # tab_rhoU = solver.mesh.cells.values[..., 1, FracMomFields.m1U]
            # tab_rhoV = solver.mesh.cells.values[..., 1, FracMomFields.m1V]
            # tab_u = solver.mesh.cells.values[..., 1, FracMomFields.U]
            # tab_v = solver.mesh.cells.values[..., 1, FracMomFields.V]

            im = ax3d.plot_surface(tabx, taby, tab_m0, cmap="plasma")

            ax3d.set_title("Fractionnal moments")
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("m0")
            ims.append([im])

        solver.step(dt)

    if plot:
        _ = ArtistAnimation(fig, ims)
        # ani.save("Advec2D_20x20.mp4", writer="ffmpeg")
        # ani.save("Advec2D_30x30.gif", writer='PillowWriter', fps=5)
        plt.show()
