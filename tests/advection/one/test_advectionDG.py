import numpy as np
import pytest

from josie.scheme.convective import ConvectiveDGScheme
from josie.bc import make_periodic, Direction
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import DGCell

from josie.general.schemes.time.rk import RK2
from josie.mesh.cellset import MeshCellSet, CellSet
from josie.state import State
from josie.solver import DGSolver
from josie.problem import Problem

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


# Advection velocity in x-direction
V = np.array([1.0, 0.0])


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

            if neighs.direction == 0:
                F[..., 0:2, 0] = (
                    self.problem.F(cells)[..., 0:2, 0]
                    + self.problem.F(neighs)[..., 2:4, 0]
                ) * 0.5 - 0.5 * (
                    self.problem.F(cells)[..., 0:2, 0]
                    - self.problem.F(neighs)[..., 2:4, 0]
                )
            if neighs.direction == 2:
                F[..., 2:4, 0] = (
                    self.problem.F(cells)[..., 2:4, 0]
                    + self.problem.F(neighs)[..., 0:2, 0]
                ) * 0.5 - 0.5 * (
                    self.problem.F(neighs)[..., 0:2, 0]
                    - self.problem.F(cells)[..., 2:4, 0]
                )
            FS = np.einsum("...ij,...j->...i", F, neighs.normals)
            return FS[..., np.newaxis]

        def CFL(
            self,
            cells: MeshCellSet,
            CFL_value: float,
        ) -> float:
            U_abs = np.linalg.norm(V)
            dx = np.min(cells.surfaces)
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
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, DGCell)
    mesh.interpolate(100, 1)
    mesh.generate()

    solver = DGSolver(mesh, Q, scheme)
    solver.scheme.alpha = 1

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]
        xc_r = np.where(xc >= 0.45)
        xc_l = np.where(xc < 0.45)
        cells.values[xc_r[0], xc_r[1], xc_r[2], :] = Q(1)
        cells.values[xc_l[0], xc_l[1], xc_l[2], :] = Q(0)

    solver.init(init_fun)

    yield solver


def test_against_real_1D(solver, plot):
    """Testing against the real 1D solver"""

    rLGLmin = 2.0
    cfl = 0.1

    tf = 0.2
    fig = plt.figure()
    ax1 = fig.add_subplot(121)

    ims = []
    x = solver.mesh.cells.centroids[..., 1, 0]

    while solver.t < tf:
        dt = solver.scheme.CFL(solver.mesh.cells, cfl * rLGLmin)

        x = solver.mesh.cells.centroids[..., 1, 0]
        u = solver.mesh.cells.values[..., 1, 0]

        if plot:
            (im1,) = ax1.plot(x, u, "ro-")
            ims.append([im1])

        solver.step(dt)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=200)
        plt.show()
