import inspect
import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.animation import ArtistAnimation

from .adv1d import main as main_1d

import josie.general.schemes.time as time_schemes
from josie.general.schemes.space import Godunov
from josie.dimension import MAX_DIMENSIONALITY
from josie.mesh.cellset import MeshCellSet
from josie.state import State
from josie.solver import Solver
from josie.problem import Problem

# Advection velocity in x-direction
V = np.array([1.0, 0.0])


class Q(State):
    fields = State.list_to_enum(["u"])  # type: ignore


def flux(state_array: Q) -> np.ndarray:
    return np.einsum("k,...ab->...abk", V, state_array)


class AdvectionProblem(Problem):
    def F(self, state_array: Q) -> np.ndarray:
        # I multiply each element of the given state array by the velocity
        # vector. I obtain an Nx2 array where each row is the flux on each
        # cell
        return flux(state_array)


@pytest.fixture(
    params=[
        member[1]
        for member in inspect.getmembers(time_schemes, inspect.isclass)
    ],
)
def scheme(request):
    class Upwind(Godunov, request.param):
        def intercellFlux(
            self, Q_L: Q, Q_R: Q, normals: np.ndarray, surfaces: np.ndarray
        ):

            nx, ny, num_dofs, num_fields = Q_L.shape

            FS = np.zeros_like(Q_L)
            F = np.zeros((nx, ny, num_dofs, num_fields, MAX_DIMENSIONALITY))

            # Dot product of each normal in `norm` by the advection velocity
            # Equivalent to: un = np.sum(Advection.V*(normals), axis=-1)
            Vn = np.einsum("...k,k->...", normals, V)

            # Check where un > 0
            idx = np.where(Vn > 0)

            if np.any(np.nonzero(idx)):
                F[idx] = flux(Q_L)[idx]

            idx = np.where(Vn < 0)
            if np.any(np.nonzero(idx)):
                F[idx] = flux(Q_R)[idx]

            FS = (
                np.einsum("...mkl,...l->...mk", F, normals)
                * surfaces[..., np.newaxis, np.newaxis]
            )

            return FS[..., np.newaxis]

        def CFL(
            self,
            cells: MeshCellSet,
            CFL_value: float,
        ) -> float:

            U_abs = np.linalg.norm(V)
            dx = np.min(cells.surfaces)

            return CFL_value * dx / U_abs

    yield Upwind(AdvectionProblem())


@pytest.fixture
def solver(scheme, mesh, init_fun, Q):
    mesh.interpolate(100, 1)
    mesh.generate()
    solver = Solver(mesh, Q, scheme)
    solver.init(init_fun)

    yield solver


def test_against_real_1D(solver, plot, tol):
    """Testing against the real 1D solver"""

    nx = solver.mesh.num_cells_x

    time, x_1d, solution = main_1d(nx, 4, 0.9, plot)
    dt = time[1] - time[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ims = []

    for i, t in enumerate(time):
        x = solver.mesh.cells.centroids[..., 0]
        x = x.reshape(x.size)
        u = solver.mesh.cells.values[..., 0]
        u = u.reshape(u.size)

        err = u - solution[i, :]

        if plot:
            (im1,) = ax1.plot(x, u, "ro-")
            (im2,) = ax1.plot(x_1d, solution[i, :], "ks-")
            ims.append([im1, im2])
            (im_err,) = ax2.plot(x_1d, err, "ks-")
            ims.append([im1, im2, im_err])

        # Check same solution with 1D-only
        # assert np.sum(err < tol) == len(x)
        solver.step(dt)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
