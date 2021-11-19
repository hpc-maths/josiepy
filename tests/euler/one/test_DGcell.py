import matplotlib.pyplot as plt
import numpy as np


from josie.bc import Dirichlet
from josie.boundary import Line
from josie.math import Direction
from josie.mesh import Mesh
from josie.mesh.cell import DGCell
from josie.mesh.cellset import MeshCellSet
from josie.euler.eos import PerfectGas
from josie.euler.exact import Exact
from josie.euler.solver import EulerSolver
from josie.euler.state import EulerState

def relative_error(a, b):
    return np.abs(a - b)


def riemann2Q(state, eos):
    """Wrap all the operations to create a complete Euler state from the
    initial Rieman Problem data
    """
    # BC
    rho = state.rho
    U = state.U
    V = state.V
    p = state.p
    rhoe = eos.rhoe(rho, p)
    e = rhoe / rho
    E = e + 0.5 * (U ** 2 + V ** 2)
    c = eos.sound_velocity(rho, p)

    return EulerState(rho, rho * U, rho * V, rho * E, rhoe, U, V, p, c, e)


def test_toro(toro_riemann_state, Scheme, plot, request):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    Q_left = riemann2Q(toro_riemann_state.left, eos)
    Q_right = riemann2Q(toro_riemann_state.right, eos)

    # Create exact Riemann solver
    riemann_solver = Exact(eos, Q_left, Q_right)

    # Solve the Riemann problem
    riemann_solver.solve()

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, DGCell)
    mesh.interpolate(100, 1)
    mesh.generate()

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]
        print(cells.centroids.shape)
        print(cells.centroids[0:2, ...])
        exit()

        cells.values[np.where(xc > 0.5), ...] = Q_right
        cells.values[np.where(xc <= 0.5), ...] = Q_left

    scheme = Scheme(eos)
    solver = EulerSolver(mesh, scheme)
    solver.init(init_fun)

    final_time = toro_riemann_state.final_time
    t = 0
    CFL = toro_riemann_state.CFL

    cells = solver.mesh.cells

    if plot:
        fig = plt.figure()
        fig.suptitle(request.node.name)
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

    # TODO: Use josie.io.strategy and josie.io.writer to save the plot every
    # time instant.  In particular it might useful to choose a Strategy (or
    # multiple strategies) and append to each strategy some "executors" that do
    # stuff with the Solver data
    while t <= final_time:
        dt = scheme.CFL(cells, CFL)

        # TODO: Basic check. The best would be to check against analytical
        # solution
        assert ~np.isnan(dt)
        solver.step(dt)

        t += dt
        print(f"Time: {t}, dt: {dt}")

    # Check that we reached the final time
    assert t >= final_time

    if plot:
        # Plot final step solution

        x = cells.centroids[..., Direction.X]
        x = x.reshape(x.size)

        rho = cells.values[..., EulerState.fields.rho]
        rho = rho.reshape(rho.size)

        U = cells.values[..., EulerState.fields.U]
        U = U.reshape(U.size)

        p = cells.values[..., EulerState.fields.p]
        p = p.reshape(p.size)

        (im1,) = ax1.plot(x, rho, "-", label="Numerical")
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\rho$")

        (im2,) = ax2.plot(x, U, "-", label="Numerical")
        ax2.set_xlabel("x")
        ax2.set_ylabel("U")

        (im3,) = ax3.plot(x, p, "-", label="Numerical")
        ax3.set_xlabel("x")
        ax3.set_ylabel("p")

        # Plot the exact solution over the final step solution
        p = []
        U = []
        rho = []
        for x_step in x:
            R = riemann_solver.sample(x_step, t)
            p.append(R[..., R.fields.p])
            U.append(R[..., R.fields.U])
            rho.append(R[..., R.fields.rho])

        (im1,) = ax1.plot(x, rho, "--", label="Exact")
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\rho$")

        (im2,) = ax2.plot(x, U, "--", label="Exact")
        ax2.set_xlabel("x")
        ax2.set_ylabel("U")

        (im3,) = ax3.plot(x, p, "--", label="Exact")
        ax3.set_xlabel("x")
        ax3.set_ylabel("p")

        # Legend
        ax1.legend()
        ax2.legend()
        ax3.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

