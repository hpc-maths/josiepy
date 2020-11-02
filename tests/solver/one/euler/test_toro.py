import inspect
import matplotlib.pyplot as plt
import numpy as np
import pytest

import josie.general.schemes.time as time_schemes

from matplotlib.animation import ArtistAnimation

from josie.bc import Dirichlet
from josie.boundary import Line
from josie.math import Direction
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.euler.eos import PerfectGas
from josie.euler.schemes import EulerScheme
from josie.euler.solver import EulerSolver
from josie.euler.state import Q


@pytest.fixture(
    params=[
        member[1]
        for member in inspect.getmembers(time_schemes, inspect.isclass)
    ],
)
def TimeScheme(request):
    yield request.param


@pytest.fixture(params=EulerScheme.__subclasses__())
def SpaceScheme(request):
    yield request.param


@pytest.fixture
def Scheme(SpaceScheme, TimeScheme):
    """ Create all the different schemes """

    class ToroScheme(SpaceScheme, TimeScheme):
        pass

    return ToroScheme


riemann_states = [
    {
        "rhoL": 1.0,
        "uL": 0.0,
        "vL": 0,
        "pL": 1.0,
        "rhoR": 0.125,
        "uR": 0,
        "vR": 0,
        "pR": 0.1,
        "t": 0.25,
        "CFL": 0.5,
    },
    {
        "rhoL": 1.0,
        "uL": -2,
        "vL": 0,
        "pL": 0.4,
        "rhoR": 1.0,
        "uR": 2.0,
        "vR": 0,
        "pR": 0.4,
        "t": 0.15,
        "CFL": 0.5,
    },
    {
        "rhoL": 1.0,
        "uL": 0,
        "vL": 0,
        "pL": 1000,
        "rhoR": 1.0,
        "uR": 0,
        "vR": 0,
        "pR": 0.01,
        "t": 0.012,
        "CFL": 0.45,
    },
    {
        "rhoL": 5.99924,
        "uL": 19.5975,
        "vL": 0,
        "pL": 460.894,
        "rhoR": 5.9924,
        "uR": -6.19633,
        "vR": 0,
        "pR": 46.0950,
        "t": 0.035,
        "CFL": 0.5,
    },
    {
        "rhoL": 1.0,
        "uL": -19.59745,
        "vL": 0,
        "pL": 1000,
        "rhoR": 1.0,
        "uR": -19.59745,
        "vR": 0,
        "pR": 0.01,
        "t": 0.035,
        "CFL": 0.5,
    },
]


@pytest.mark.parametrize("riemann", riemann_states)
def test_toro(riemann, Scheme, plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    # BC
    rhoL = riemann["rhoL"]
    uL = riemann["uL"]
    vL = riemann["vL"]
    pL = riemann["pL"]
    rhoeL = eos.rhoe(rhoL, pL)
    EL = rhoeL / rhoL + 0.5 * (uL ** 2 + vL ** 2)
    cL = eos.sound_velocity(rhoL, pL)

    rhoR = riemann["rhoR"]
    uR = riemann["uR"]
    vR = riemann["vR"]
    pR = riemann["pR"]
    rhoeR = eos.rhoe(rhoR, pR)
    ER = rhoeR / rhoR + 0.5 * (uR ** 2 + vR ** 2)
    cR = eos.sound_velocity(rhoR, pR)

    Q_left = Q(rhoL, rhoL * uL, rhoL * vL, rhoL * EL, rhoeL, uL, vL, pL, cL)
    Q_right = Q(rhoR, rhoR * uR, rhoR * vR, rhoR * ER, rhoeR, uR, vR, pR, cR)

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(500, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]

        cells.values[np.where(xc > 0.5), ...] = Q_right
        cells.values[np.where(xc <= 0.5), ...] = Q_left

    scheme = Scheme(eos)
    solver = EulerSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = riemann["t"]
    t = 0
    CFL = riemann["CFL"]

    if plot:
        ims = []
        fig = plt.figure()
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

    # TODO: Use josie.io.strategy and josie.io.writer to save the plot every
    # time instant.  In particular it might useful to choose a Strategy (or
    # multiple strategies) and append to each strategy some "executors" that do
    # stuff with the Solver data
    while t <= final_time:
        cells = solver.mesh.cells

        x = cells.centroids[..., Direction.X]
        x = x.reshape(x.size)

        rho = cells.values[..., Q.fields.rho]
        rho = rho.reshape(rho.size)

        U = cells.values[..., Q.fields.U]
        U = U.reshape(U.size)

        p = cells.values[..., Q.fields.p]
        p = p.reshape(p.size)

        if plot:
            (im1,) = ax1.plot(x, rho, "k-")
            ax1.set_xlabel("x")
            ax1.set_ylabel(r"$\rho$")

            (im2,) = ax2.plot(x, U, "k-")
            ax2.set_xlabel("x")
            ax2.set_ylabel("U")

            (im3,) = ax3.plot(x, p, "k-")
            ax3.set_xlabel("x")
            ax3.set_ylabel("p")

            ims.append([im1, im2, im3])

        dt = scheme.CFL(cells, CFL)

        # TODO: Basic check. The best would be to check against analytical
        # solution
        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50, repeat=False)
        plt.tight_layout()
        plt.show()
        plt.close()
