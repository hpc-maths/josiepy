import pytest

from josie.bc import BoundaryCondition, Dirichlet, Neumann
from josie.boundary import Line
from josie.euler.eos import PerfectGas
from josie.euler.schemes import Rusanov
from josie.general.schemes.time import ExplicitEuler
from josie.io.write.writer import XDMFWriter
from josie.io.write.strategy import TimeStrategy
from josie.ns.schemes.scheme import NSScheme
from josie.ns.schemes.diffusive import CentralDifferenceGradient
from josie.ns.solver import NSSolver
from josie.ns.state import NSState
from josie.ns.transport import NSConstantTransport
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell


def inlet_state():
    rho = 1
    U = 1
    V = 0
    e = 300

    eos = PerfectGas()
    p = eos.p(rho, e)
    E = e + 0.5 * (U ** 2 + V ** 2)
    c = eos.sound_velocity(rho, p)
    inlet_state = NSState(rho, rho * U, rho * V, rho * E, rho * e, U, V, p, c)

    return inlet_state


class Wall(BoundaryCondition):
    def __init__(self, eos=PerfectGas()):
        bc_state = NSState(
            rho=Neumann(0),
            rhoU=Dirichlet(0),
            rhoV=Dirichlet(0),
            rhoE=Neumann(0),
            rhoe=Neumann(0),
            U=Dirichlet(0),
            V=Dirichlet(0),
            p=Neumann(0),
            c=Neumann(0),
        )

        super().__init__(bc_state)


class Outflow(BoundaryCondition):
    def __init__(self, eos=PerfectGas()):
        bc_state = NSState(
            rho=Neumann(0),
            rhoU=Neumann(0),
            rhoV=Neumann(0),
            rhoE=Neumann(0),
            rhoe=Neumann(0),
            U=Neumann(0),
            V=Neumann(0),
            p=Neumann(0),
            c=Neumann(0),
        )

        super().__init__(bc_state)


def init_fun(cells):
    cells.values[:] = inlet_state()


@pytest.fixture
def boundaries():
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [3, 0])
    right = Line([3, 0], [3, 1])
    top = Line([0, 1], [3, 1])

    left.bc = Dirichlet(inlet_state())
    right.bc = Outflow()
    bottom.bc = Wall()
    top.bc = Wall()

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(99, 33)
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

    class Scheme(NSScheme, CentralDifferenceGradient, Rusanov, ExplicitEuler):
        pass

    yield Scheme(eos, transport)


@pytest.fixture
def solver(mesh, Q, scheme):
    """ A dummy solver instance with initiated state """

    solver = NSSolver(mesh, scheme)
    solver.init(init_fun)

    yield solver


def test_poiseille(solver, plot):
    if plot:
        solver.plot()

    final_time = 2.5
    CFL = 0.5

    write_strategy = TimeStrategy(dt_save=0.05)
    writer = XDMFWriter("ns.xdmf", write_strategy, solver, final_time, CFL)
    writer.solve()

    if plot:
        solver.show("U")
