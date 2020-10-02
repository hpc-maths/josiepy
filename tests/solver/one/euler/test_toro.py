import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.animation import ArtistAnimation

from josie.bc import Dirichlet
from josie.geom import Line
from josie.mesh import Mesh, SimpleCell
from josie.euler.eos import PerfectGas
from josie.euler.schemes import Rusanov
from josie.general.schemes.time import ExplicitEuler
from josie.euler.solver import EulerSolver
from josie.euler.state import Q


class ToroScheme(Rusanov, ExplicitEuler):
    pass


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
def test_toro(riemann, plot):
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

    def init_fun(solver: EulerSolver):
        xc = solver.mesh.centroids[:, :, 0]

        solver.values[np.where(xc > 0.5), :, :] = Q_right
        solver.values[np.where(xc <= 0.5), :, :] = Q_left

    scheme = ToroScheme(eos)
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

    while t <= final_time:
        x = solver.mesh.centroids[:, :, 0]
        x = x.reshape(x.size)

        rho = solver.values[:, :, 0]
        rho = rho.reshape(rho.size)

        U = solver.values[:, :, 5]
        U = U.reshape(U.size)

        p = solver.values[:, :, 7]
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

        dt = scheme.CFL(
            solver.values, solver.mesh.volumes, solver.mesh.surfaces, CFL
        )
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50, repeat=False)
        plt.tight_layout()
        plt.show()
        plt.close()
