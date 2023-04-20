# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from josie.boundary import Line
from josie.euler.eos import PerfectGas
from josie.euler.schemes import Rusanov
from josie.math import Direction
from josie.general.schemes.time import ExplicitEuler
from josie.io.write.writer import XDMFWriter
from josie.io.write.strategy import TimeStrategy
from josie.ns.bc import Inlet, Outflow, NoSlip
from josie.ns.schemes.scheme import NSScheme
from josie.ns.schemes.diffusive import CentralDifferenceGradient
from josie.ns.solver import NSSolver
from josie.ns.state import NSState
from josie.ns.transport import NSConstantTransport
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.general.schemes.space import Godunov


@pytest.fixture
def eos():
    return PerfectGas()


@pytest.fixture
def init_state(eos):
    rho = 1
    U = 1
    V = 0
    e = 300

    p = eos.p(rho, e)
    E = e + 0.5 * (U**2 + V**2)
    c = eos.sound_velocity(rho, p)
    init_state = NSState(rho, rho * U, rho * V, rho * E, rho * e, U, V, p, c, e)

    yield init_state


@pytest.fixture
def U_inlet():
    def U_inlet_fun(cells, t):
        nx, _, _ = cells.centroids.shape
        y = cells.centroids[..., Direction.Y]

        U = -20 * ((y - 0.5) ** 4) + 1.25

        return U

    yield U_inlet_fun


@pytest.fixture
def init_fun(init_state):
    def _init_fun(cells):
        cells.values[:] = init_state

    yield _init_fun


@pytest.fixture
def boundaries(init_state, U_inlet, eos):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [3, 0])
    right = Line([3, 0], [3, 1])
    top = Line([0, 1], [3, 1])

    Q_init: NSState = init_state

    V_inlet = Q_init[Q_init.fields.V]
    rhoe_inlet = Q_init[Q_init.fields.rhoe]
    rho_inlet = Q_init[Q_init.fields.rho]
    e_inlet = rhoe_inlet / rho_inlet
    p_inlet = Q_init[Q_init.fields.p]

    left.bc = Inlet(U_inlet, V_inlet, e_inlet, eos)
    right.bc = Outflow(p_inlet, eos)
    bottom.bc = NoSlip(eos)
    top.bc = NoSlip(eos)

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(150, 50)
    mesh.generate()

    yield mesh


@pytest.fixture
def scheme():
    mu = 1.8e-5
    # lmbda = -2 / 3 * mu
    lmbda = 0
    alphaT = 2.1e-5

    transport = NSConstantTransport(
        viscosity=mu, bulk_viscosity=lmbda, thermal_diffusivity=alphaT
    )

    eos = PerfectGas()

    class Scheme(Godunov, NSScheme, CentralDifferenceGradient, Rusanov, ExplicitEuler):
        pass

    yield Scheme(eos, transport)


@pytest.fixture
def solver(mesh, Q, scheme, init_fun):
    """A dummy solver instance with initiated state"""

    solver = NSSolver(mesh, scheme)
    solver.init(init_fun)

    yield solver


def test_poiseuille(solver, plot):
    if plot:
        solver.plot()

    final_time = 1
    CFL = 0.5

    write_strategy = TimeStrategy(dt_save=0.05, animate=True)
    writer = XDMFWriter("ns.xdmf", write_strategy, solver, final_time, CFL)
    writer.solve()

    if plot:
        solver.show("U")
