import numpy as np
import pytest

from josie.bc import make_periodic, Direction
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import DGCell
from josie.scheme.convective import ConvectiveDGScheme

from josie.general.schemes.time.rk import RK2
from josie.mesh.cellset import MeshCellSet, CellSet
from josie.state import State
from josie.solver import DGSolver
from josie.problem import Problem

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# Advection velocity in xy-direction
V = np.array([1.0, 1.0])


class Q(State):
    fields = State.list_to_enum(["u"])  # type: ignore


class AdvectionProblem(Problem):
    def __init__(self, V: np.ndarray):
        self.V = V

    def F(self, cells: MeshCellSet) -> np.ndarray:
        # I multiply each element of the given state array by the velocity
        # vector. I obtain an Nx2 array where each row is the flux on each
        # cell
        return np.einsum("j,...i->...j", self.V, cells.values)


class SchemeAdvDG(ConvectiveDGScheme):
    uavg: np.ndarray
    minu: np.ndarray
    maxu: np.ndarray
    theta: np.ndarray

    def __init__(self, problem: AdvectionProblem):
        super().__init__(problem)

    def post_init(self, cells: MeshCellSet):
        super().post_init(cells)
        nx, ny, _, nfields = cells.values.shape

        self.uavg = np.zeros((nx, ny, 1, nfields))
        self.minu = np.zeros((nx, ny, 1, nfields))
        self.maxu = np.zeros((nx, ny, 1, nfields))
        self.theta = np.zeros((nx, ny, 1, nfields))

    def stiffness_fluxes(self, cells: MeshCellSet):
        vec = np.einsum(
            "ijk,...jk->...ik",
            self.K_ref,
            self.problem.F(cells),
        )

        return (2.0 / self.dx) * vec[..., [0]] + (2.0 / self.dy) * vec[
            ..., [1]
        ]

    def post_integrate_fluxes(self, cells: MeshCellSet):
        super().post_integrate_fluxes(cells)
        self.limiter(cells)

    def limiter(self, cells: MeshCellSet):
        uavg = self.uavg
        minu = self.minu
        maxu = self.maxu
        theta = self.theta

        uavg = np.mean(cells.values, axis=-2, keepdims=True)
        umin = 0.0
        umax = 1.0
        minu = np.amin(cells.values, axis=-2, keepdims=True)
        maxu = np.amax(cells.values, axis=-2, keepdims=True)

        theta = np.where(
            ((maxu - uavg) != 0) * ((minu - uavg) != 0),
            np.minimum(
                np.abs((umax - uavg) / (maxu - uavg)),
                np.abs((umin - uavg) / (minu - uavg)),
            ),
            0,
        )
        theta = np.minimum(1, theta)

        cells.values = theta * (cells.values - uavg) + uavg


@pytest.fixture
def scheme():
    class Upwind(SchemeAdvDG, RK2):
        def F(self, cells: MeshCellSet, neighs: CellSet):
            values = cells.values
            nx, ny, num_dofs, _ = values.shape

            FS = np.zeros_like(values)
            F = np.zeros((nx, ny, num_dofs, 2))

            maxvel = np.amax(np.fabs(V))

            if neighs.direction == 0:
                F[..., 0:2, 0] = (
                    self.problem.F(cells)[..., 0:2, 0]
                    + self.problem.F(neighs)[..., 2:4, 0]
                ) * 0.5 - 0.5 * maxvel * (
                    self.problem.F(cells)[..., 0:2, 0]
                    - self.problem.F(neighs)[..., 2:4, 0]
                )
            if neighs.direction == 1:
                F[..., 0, 1] = (
                    self.problem.F(cells)[..., 0, 1]
                    + self.problem.F(neighs)[..., 1, 1]
                ) * 0.5 - 0.5 * maxvel * (
                    self.problem.F(cells)[..., 0, 1]
                    - self.problem.F(neighs)[..., 1, 1]
                )
                F[..., 2, 1] = (
                    self.problem.F(cells)[..., 2, 1]
                    + self.problem.F(neighs)[..., 3, 1]
                ) * 0.5 - 0.5 * maxvel * (
                    self.problem.F(cells)[..., 2, 1]
                    - self.problem.F(neighs)[..., 3, 1]
                )
            if neighs.direction == 2:
                F[..., 2:4, 0] = (
                    self.problem.F(cells)[..., 2:4, 0]
                    + self.problem.F(neighs)[..., 0:2, 0]
                ) * 0.5 - 0.5 * maxvel * (
                    self.problem.F(neighs)[..., 0:2, 0]
                    - self.problem.F(cells)[..., 2:4, 0]
                )
            if neighs.direction == 3:
                F[..., 1, 1] = (
                    self.problem.F(cells)[..., 1, 1]
                    + self.problem.F(neighs)[..., 0, 1]
                ) * 0.5 - 0.5 * maxvel * (
                    self.problem.F(neighs)[..., 0, 1]
                    - self.problem.F(cells)[..., 1, 1]
                )
                F[..., 3, 1] = (
                    self.problem.F(cells)[..., 3, 1]
                    + self.problem.F(neighs)[..., 2, 1]
                ) * 0.5 - 0.5 * maxvel * (
                    self.problem.F(neighs)[..., 2, 1]
                    - self.problem.F(cells)[..., 3, 1]
                )
            FS = np.einsum("...ij,...j->...i", F, neighs.normals)
            return FS[..., np.newaxis]

        def CFL(
            self,
            cells: MeshCellSet,
            CFL_value: float,
        ) -> float:
            U_abs = np.linalg.norm(V)
            dx = cells.min_length
            return CFL_value * dx / U_abs

    yield Upwind(AdvectionProblem(V))


@pytest.fixture
def solver(scheme, Q):
    """1D problem along x"""
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left, right = make_periodic(left, right, Direction.X)
    bottom, top = make_periodic(bottom, top, Direction.Y)

    mesh = Mesh(left, bottom, right, top, DGCell)
    mesh.interpolate(30, 30)
    mesh.generate()

    solver = DGSolver(mesh, Q, scheme)
    solver.scheme.alpha = 1

    def init_fun(cells: MeshCellSet):
        c_x = cells.centroids[..., 0]
        c_y = cells.centroids[..., 1]
        cells.values[..., 0] = np.where(
            (np.abs(c_x - 0.5) < 0.1) * (np.abs(c_y - 0.5) < 0.1), 1.0, 0.0
        )

    solver.init(init_fun)

    yield solver


def test_advection(solver, plot):
    """Testing against the real 1D solver"""

    rLGLmin = 2.0
    cfl = 0.1

    tf = 0.5

    fig = plt.figure()
    ax3d = plt.axes(projection="3d")
    ims = []

    while solver.t < tf:
        dt = solver.scheme.CFL(solver.mesh.cells, cfl * rLGLmin)
        solver.step(dt)

        if plot:
            tabx = solver.mesh.cells.centroids[..., 1, 0]
            taby = solver.mesh.cells.centroids[..., 1, 1]
            tabu = solver.mesh.cells.values[..., 1, 0]

            im = ax3d.plot_surface(tabx, taby, tabu, cmap="plasma")

            ax3d.set_title("2D Advection equation")
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("U")
            ims.append([im])

    if plot:
        _ = ArtistAnimation(fig, ims)
        # ani.save("Advec2D_20x20.mp4", writer="ffmpeg")
        # ani.save("Advec2D_30x30.gif", writer='PillowWriter', fps=5)
        plt.show()
