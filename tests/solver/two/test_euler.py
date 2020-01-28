import numpy as np
import pytest

from josie.bc import Dirichlet, Neumann, Direction, make_periodic
from josie.geom import Line
from josie.mesh import Mesh, SimpleCell
from josie.solver.euler import Rusanov, Q, EulerSolver, PerfectGas

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
    },
]


def periodic(bottom, top):
    return make_periodic(bottom, top, Direction.Y)


def neumann(bottom, top):
    top.bc = Neumann(Q.zeros())
    bottom.bc = Neumann(Q.zeros())

    return bottom, top


@pytest.mark.parametrize("riemann_problem", riemann_states)
@pytest.mark.parametrize("vertical_bc_fun", [periodic, neumann])
def test_toro(riemann_problem, vertical_bc_fun, plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    # BC
    rhoL = riemann_problem["rhoL"]
    uL = riemann_problem["uL"]
    vL = riemann_problem["vL"]
    pL = riemann_problem["pL"]
    rhoeL = eos.rhoe(rhoL, pL)
    EL = rhoeL / rhoL + 0.5 * (uL ** 2 + vL ** 2)
    cL = eos.sound_velocity(rhoL, pL)

    rhoR = riemann_problem["rhoR"]
    uR = riemann_problem["uR"]
    vR = riemann_problem["vR"]
    pR = riemann_problem["pR"]
    rhoeR = eos.rhoe(rhoR, pR)
    ER = rhoeR / rhoR + 0.5 * (uR ** 2 + vR ** 2)
    cR = eos.sound_velocity(rhoR, pR)

    Q_left = Q(rhoL, rhoL * uL, rhoL * vL, rhoL * EL, rhoeL, uL, vL, pL, cL)
    Q_right = Q(rhoR, rhoR * uR, rhoR * vR, rhoR * ER, rhoeR, uR, vR, pR, cR)

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    bottom, top = vertical_bc_fun(bottom, top)

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(100, 100)
    mesh.generate()

    def init_fun(solver: EulerSolver):
        xc = solver.mesh.centroids[:, :, 0]

        idx_left = np.where(xc >= 0.5)
        idx_right = np.where(xc < 0.5)

        solver.values[idx_left[0], idx_right[1], :] = Q_right
        solver.values[idx_right[0], idx_right[1], :] = Q_left

    solver = EulerSolver(mesh, eos)
    solver.init(init_fun)

    final_time = 0.25
    t = 0
    CFL = 0.80
    rusanov = Rusanov()

    while t <= final_time:
        if plot:
            solver.animate(t)
            solver.save(t, "toro.xmf")

        dt = rusanov.CFL(
            solver.values,
            solver.mesh.volumes,
            solver.mesh.normals,
            solver.mesh.surfaces,
            CFL,
        )
        solver.step(dt, rusanov)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        solver.show("U")
