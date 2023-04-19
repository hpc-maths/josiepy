import abc
import numpy as np
import pytest
import scipy.integrate as integrate

from josie.scheme.scheme import Scheme
from josie.bc import make_periodic, Direction
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import DGCell

from josie.general.schemes.time.rk import RK, ButcherTableau
from josie.mesh.cellset import MeshCellSet, CellSet, NeighboursCellSet
from josie.state import State
from josie.solver import Solver
from josie.problem import Problem

import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.animation import ArtistAnimation

from typing import NoReturn, Callable

# Advection velocity in x-direction
V = np.array([1.0, 0.0])


class Q(State):
    fields = State.list_to_enum(["u"])  # type: ignore


class RKDG(RK):
    def post_init(self, cells: MeshCellSet):
        r"""A Runge-Kutta method needs to store intermediate steps. It needs
        :math:`s - 1` additional storage slots, where :math:`s` is the number
        of steps of the Runge-Kutta method

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells
        """

        super().post_init(cells)

        nx, ny, num_dofs, num_fields = cells.values.shape

        self._ks: np.ndarray = np.empty(
            (nx, ny, num_dofs, num_fields, self.num_steps - 1)
        )

    def k(self, mesh: Mesh, dt: float, t: float, step: int):
        r"""Recursive function that computes all the :math:`k_s` coefficients
        from :math:`s = 0` to :math:`s = \text{step}`

        The highest :math:`k_s` value is stored in :attr:`_fluxes`
        """
        if step > 0:
            self.k(mesh, dt, t, step - 1)
            self._ks[..., step - 1] = self._fluxes.copy()
            self._fluxes.fill(0)

        c = self.butcher.c_s[step]
        a_s = self.butcher.a_s[step : 2 * step + 1]

        t += c * dt
        step_cells = mesh.cells.copy()
        step_cells.values -= dt * np.einsum(
            "...i,...j->...", a_s, self._ks[..., :step]
        )

        step_cells.update_ghosts(mesh.boundaries, t)

        self.pre_accumulate(step_cells, t)

        for neighs in step_cells.neighbours:
            self.accumulate(step_cells, neighs, t)

        vec = np.einsum(
            "ijk,...jk->...ik",
            self.K_ref,
            self.problem.F(step_cells.values),
        )
        dx = mesh._x[1, 0] - mesh._x[0, 0]
        dy = mesh._y[0, 1] - mesh._y[0, 0]
        vec2 = (2.0 / dx) * vec[..., [0]] + (2.0 / dy) * vec[..., [1]]

        self._fluxes -= vec2
        self._fluxes = np.linalg.solve(self.M_ref, self._fluxes)


class RKDG2Alpha(RKDG):
    def __init__(self, problem: Problem, alpha: float):
        self.alpha = alpha

        butcher = ButcherTableau(
            a_s=np.array([alpha]),
            b_s=np.array([1 - 1 / (2 * alpha), 1 / (2 * alpha)]),
            c_s=np.array([alpha]),
        )

        super().__init__(problem, butcher)


class RKDG2(RKDG2Alpha):
    r"""Implements the explicit 2nd-order Runge-Kutta scheme with :math:`\alpha =
    2/3`
    """

    time_order: float = 2

    # def __init__(self, problem: Problem):
    #    super().__init__(problem, 2 / 3)

    def __init__(self, problem: Problem):
        super().__init__(problem, 1)


class SolverDG(Solver):
    def init(self, init_fun: Callable[[MeshCellSet], NoReturn]):
        super().init(init_fun)

        # Init a local mass matrix in the element of reference
        # Dim : (num_dof, num_dof)
        self.scheme.M_ref = self.mesh.cell_type.refMass()
        # Init a local stiffness matrix in the element of reference
        # Dim : (num_dof, num_dof)
        self.scheme.K_ref = self.mesh.cell_type.refStiff()

        # Init a local edge-mass matrix in the element of reference
        # One matrix for each direction
        # Dim : (num_dof, num_dof)
        self.scheme.eM_ref_tab = self.mesh.cell_type.refMassEdge()

        # Init jacobians
        self.scheme.J = self.jacob(self.mesh)
        # Init edge jacobians
        # One for each direction
        self.scheme.eJ = self.jacob1D(self.mesh)

    def jacob(self, mesh):
        x = mesh._x
        dx = mesh._x[1, 0] - mesh._x[0, 0]
        dy = mesh._y[0, 1] - mesh._y[0, 0]

        # Works only for structured mesh (no rotation, only x-axis
        # and/or y-axis stretch)
        # self.jac = J^{-1}
        self.jac = (4.0 / (dx * dy)) * np.ones(x[1:, :-1].shape)
        return self.jac

    def jacob1D(self, mesh):
        self.jacEdge = mesh.cells.surfaces / 2

        return self.jacEdge


class AdvectionProblem(Problem):
    def F(self, state_array: Q) -> np.ndarray:
        # I multiply each element of the given state array by the velocity
        # vector. I obtain an Nx2 array where each row is the flux on each
        # cell
        return flux(state_array)


class SchemeAdvDG(Scheme):
    problem: AdvectionProblem
    M_ref: np.ndarray
    eM_ref_tab: np.ndarray
    J: np.ndarray
    eJ: np.ndarray
    K_ref: np.ndarray

    def __init__(self):
        super().__init__(AdvectionProblem())

    def accumulate(
        self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float
    ):
        # Compute fluxes computed eventually by the other terms (diffusive,
        # nonconservative, source)
        super().accumulate(cells, neighs, t)
        # Add conservative contribution
        self._fluxes += np.einsum(
            "...,...,ij,...jk->...ik",
            self.eJ[..., neighs.direction],
            self.J,
            self.eM_ref_tab[neighs.direction],
            self.F(cells, neighs),
        )

    @abc.abstractmethod
    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet) -> State:
        raise NotImplementedError


def flux(state_array: Q) -> np.ndarray:
    return np.einsum("j,...i->...j", V, state_array)


def upwind(cells: MeshCellSet, neighs: CellSet):
    values = cells.values
    nx, ny, num_dofs, _ = values.shape

    FS = np.zeros_like(values)
    F = np.zeros((nx, ny, num_dofs, 2))

    # I do a dot product of each normal in `norm` by the advection velocity
    # Equivalent to: un = np.sum(Advection.V*(normals), axis=-1)

    if neighs.direction == 0:
        F[..., 0:2, 0] = (
            flux(values)[..., 0:2, 0] + flux(neighs.values)[..., 2:4, 0]
        ) * 0.5 - 0.5 * (flux(values)[..., 0:2, 0] - flux(neighs.values)[..., 2:4, 0])
    if neighs.direction == 2:
        F[..., 2:4, 0] = (
            flux(values)[..., 2:4, 0] + flux(neighs.values)[..., 0:2, 0]
        ) * 0.5 - 0.5 * (flux(neighs.values)[..., 0:2, 0] - flux(values)[..., 2:4, 0])
    FS = np.einsum("...ij,...j->...i", F, neighs.normals)
    return FS[..., np.newaxis]


@pytest.fixture
def scheme():
    class Upwind(SchemeAdvDG, RKDG2):
        def F(self, cells: MeshCellSet, neighs: CellSet):
            return upwind(cells, neighs)

        def CFL(
            self,
            cells: MeshCellSet,
            CFL_value: float,
        ) -> float:
            U_abs = np.linalg.norm(V)
            dx = np.min(cells.surfaces)
            return CFL_value * dx / U_abs

    yield Upwind()


def door(x, t):
    return (x > 0.2 + V[0] * t) * (x < 0.4 + V[0] * t) * 1


def bump(x, t):
    b = 0.1
    a = 0.3 + V[0] * t
    return np.where(
        np.abs(x - a) < b - 1e-8,
        np.exp(b**2 / ((x - a) ** 2 - b**2)),
        0,
    )


def const(x, t):
    return np.ones_like(x)


@pytest.fixture(params=sorted([const, bump, door], key=lambda c: c.__name__))
def u_exa(request):
    yield request.param


@pytest.fixture
def solver(scheme, Q, u_exa):
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

    solver = SolverDG(mesh, Q, scheme)

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0, :, 0]
        u_init = u_exa(xc, 0)
        cells.values[..., 0, :, 0] = u_init.view(Q)

    solver.init(init_fun)

    yield solver


def test_against_real_1D(solver, plot, tol, u_exa):
    """Testing against the real 1D solver"""

    rLGLmin = 2.0
    cfl = 0.1
    dx = solver.mesh._x[1, 0] - solver.mesh._x[0, 0]
    dt = cfl * rLGLmin * dx / np.linalg.norm(V)

    tf = 0.2
    time = np.arange(0, tf, dt)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_ylim([-0.3, 1.3])

    ims = []
    err = 0

    for _, t in enumerate(time):
        x = solver.mesh.cells.centroids[..., 0, :, 0]
        u = solver.mesh.cells.values

        if plot:
            lines = np.asarray(
                # BUG: the linear approx should be reversed LEFT/RIGHT
                [
                    [
                        [x[i, 0], u[i, 0, 2, 0]],
                        [x[i, 2], u[i, 0, 0, 0]],
                    ]
                    for i in range(x.shape[0])
                ]
            )
            (im1,) = ax1.plot(x[..., 0], u_exa(x[..., 0], t), ":k")
            lc = mc.LineCollection(lines, linewidths=1)
            DG_lines = ax1.add_collection(lc)
            ims.append([DG_lines, im1])

        solver.step(dt)

    err = sum(
        [
            integrate.quad(
                lambda xx: (
                    np.interp(
                        xx,
                        [x[i, 0], x[i, 2]],
                        [u[i, 0, 2, 0], u[i, 0, 0, 0]],
                    )
                    - u_exa(xx, t)
                )
                ** 2,
                x[i, 0],
                x[i, 2],
            )[0]
            for i in range(x.shape[0])
        ]
    )
    print(np.sqrt(err))

    if plot:
        _ = ArtistAnimation(fig, ims, interval=200)
        plt.show()
