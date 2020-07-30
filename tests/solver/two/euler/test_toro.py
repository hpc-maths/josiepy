import numpy as np
import pytest

from josie.bc import Dirichlet, Neumann, Direction, make_periodic
from josie.euler.eos import PerfectGas
from josie.euler.schemes import Rusanov
from josie.euler.solver import EulerSolver
from josie.euler.state import Q
from josie.general.schemes.time import ExplicitEuler
from josie.geom import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet


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
        "CFL": 0.4,
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
        "CFL": 0.4,
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
        "CFL": 0.4,
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
        "CFL": 0.4,
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
        "CFL": 0.4,
    },
]


def init_test(direction, riemann_problem, bc_fun):
    """A handy function to init the test state on the base of the direction,
    to avoid code duplication"""

    if direction is Direction.X:
        uL = riemann_problem["uL"]
        vL = riemann_problem["vL"]

        uR = riemann_problem["uR"]
        vR = riemann_problem["vR"]

    elif direction is Direction.Y:
        uL = riemann_problem["vL"]
        vL = riemann_problem["uL"]

        uR = riemann_problem["vR"]
        vR = riemann_problem["uR"]

    # Common stuff in all directions
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    rhoL = riemann_problem["rhoL"]
    pL = riemann_problem["pL"]
    rhoeL = eos.rhoe(rhoL, pL)
    EL = rhoeL / rhoL + 0.5 * (uL ** 2 + vL ** 2)
    cL = eos.sound_velocity(rhoL, pL)

    rhoR = riemann_problem["rhoR"]
    pR = riemann_problem["pR"]
    rhoeR = eos.rhoe(rhoR, pR)
    ER = rhoeR / rhoR + 0.5 * (uR ** 2 + vR ** 2)
    cR = eos.sound_velocity(rhoR, pR)

    Q_left = Q(rhoL, rhoL * uL, rhoL * vL, rhoL * EL, rhoeL, uL, vL, pL, cL)
    Q_right = Q(rhoR, rhoR * uR, rhoR * vR, rhoR * ER, rhoeR, uR, vR, pR, cR)

    if direction is Direction.X:
        left.bc = Dirichlet(Q_left)
        right.bc = Dirichlet(Q_right)
        bottom, top = bc_fun(bottom, top, Direction.Y)

        def init_fun(cells: MeshCellSet):
            xc = cells.centroids[..., 0]

            idx_left = np.where(xc >= 0.5)
            idx_right = np.where(xc < 0.5)

            cells.values[idx_left[0], idx_left[1], :] = Q_right
            cells.values[idx_right[0], idx_right[1], :] = Q_left

        plot_var = "U"

    elif direction is Direction.Y:
        bottom.bc = Dirichlet(Q_left)
        top.bc = Dirichlet(Q_right)
        left, right = bc_fun(left, right, Direction.X)

        def init_fun(cells: MeshCellSet):
            yc = cells.centroids[..., 1]

            idx_top = np.where(yc >= 0.5)
            idx_btm = np.where(yc < 0.5)

            cells.values[idx_btm[0], idx_btm[1], ...] = Q_left
            cells.values[idx_top[0], idx_top[1], ...] = Q_right

        plot_var = "V"

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(30, 30)
    mesh.generate()

    scheme = ToroScheme(eos)
    solver = EulerSolver(mesh, scheme)
    solver.init(init_fun)

    return solver, plot_var


def neumann(first, second, direction):
    second.bc = Neumann(np.zeros(len(Q.fields)).view(Q))
    first.bc = Neumann(np.zeros(len(Q.fields)).view(Q))

    return first, second


def periodic(first, second, direction):
    return make_periodic(first, second, direction)


@pytest.mark.parametrize("riemann_problem", riemann_states)
@pytest.mark.parametrize("bc_fun", [periodic, neumann])
@pytest.mark.parametrize("direction", [Direction.X, Direction.Y])
def test_toro(direction, riemann_problem, bc_fun, plot):

    solver, plot_var = init_test(direction, riemann_problem, bc_fun)
    scheme = solver.scheme

    if plot:
        solver.plot()

    final_time = 0.25
    t = 0
    CFL = riemann_problem["CFL"]

    while t <= final_time:
        if plot:
            solver.animate(t)
            # solver.save(t, "toro.xmf")

        dt = scheme.CFL(
            solver.mesh.cells,
            CFL,
        )
        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    if plot:
        solver.show(plot_var)
