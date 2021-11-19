import inspect
import abc
import matplotlib
import matplotlib.pyplot as plt

from josie.scheme.scheme import Scheme

matplotlib.use("TkAgg")
import numpy as np
import pytest
import math

from josie.bc import make_periodic, Direction
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import DGCell


from matplotlib.animation import ArtistAnimation

from .adv1d import main as main_1d

import josie.general.schemes.time as time_schemes
from josie.general.schemes.time.rk import RK, ButcherTableau
from josie.mesh.cellset import MeshCellSet, CellSet
from josie.state import State
from josie.solver import Solver
from josie.problem import Problem

from typing import NoReturn, Callable
import ipdb

# from josie.scheme.convective import ConvectiveScheme
from josie.mesh.cellset import NeighboursCellSet, MeshCellSet


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
        step_cells.values -= dt * np.einsum("...i,...j->...", a_s, self._ks[..., :step])

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
        y = mesh._y
        dx = mesh._x[1, 0] - mesh._x[0, 0]
        dy = mesh._y[0, 1] - mesh._y[0, 0]

        # Works only for structured mesh (no rotation, only x-axis and/or y-axis stretch)
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

    def accumulate(self, cells: MeshCellSet, neighs: NeighboursCellSet, t: float):

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
    Vn = np.einsum("...k,k->...", neighs.normals, V)

    # Check where un > 0
    idx = np.where(Vn > 0)
    if np.any(np.nonzero(idx)):
        F[idx] = flux(values)[idx]
    idx = np.where(Vn < 0)
    if np.any(np.nonzero(idx)):
        F[idx] = flux(neighs.values)[idx]

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


@pytest.fixture
def solver(scheme, Q):
    """1D problem along x"""
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [2, 0])
    right = Line([2, 0], [2, 1])
    top = Line([0, 1], [2, 1])

    # left = Line([-math.pi, 0], [-math.pi, 1])
    # bottom = Line([-math.pi, 0], [math.pi, 0])
    # right = Line([math.pi, 0], [math.pi, 1])
    # top = Line([-math.pi, 1], [math.pi, 1])

    # left = Line([0, 0], [0, 1])
    # bottom = Line([0, 0], [2 * math.pi, 0])
    # right = Line([2 * math.pi, 0], [2 * math.pi, 1])
    # top = Line([0, 1], [2 * math.pi, 1])

    left, right = make_periodic(left, right, Direction.X)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, DGCell)
    mesh.interpolate(100, 1)
    mesh.generate()

    solver = SolverDG(mesh, Q, scheme)

    def init_fun(cells: MeshCellSet):

        xc = cells.centroids[..., 0]
        # sigma = 125.0 * 1000.0 / (33.0 ** 2)
        # c = 0.5
        # for i in range(mesh.num_cells_x):
        #    for j in range(mesh.num_cells_y):
        #        for k in range(mesh.cell_type.num_dofs):
        #            cells.values[i, j, k, :] = math.exp(
        #                -sigma * math.fabs(xc[i, j, k] - c) ** 2
        #            )

        # for i in range(mesh.num_cells_x):
        #    for j in range(mesh.num_cells_y):
        #        for k in range(mesh.cell_type.num_dofs):
        #            cells.values[i, j, k, :] = math.sin(xc[i, j, k])

        xc_r = np.where(xc >= 0.45)
        xc_l = np.where(xc < 0.45)
        cells.values[xc_r[0], xc_r[1], xc_r[2], :] = Q(1)
        # cells.values[xc_l[0], xc_l[1], xc_l[2], :] = Q(1)
        cells.values[xc_l[0], xc_l[1], xc_l[2], :] = Q(0)

    solver.init(init_fun)

    yield solver


def test_against_real_1D(solver, plot, tol):
    """Testing against the real 1D solver"""

    cfl = 2.0 / 3
    dx = 1 / solver.mesh.num_cells_x
    dt = cfl * dx / np.linalg.norm(V)

    tf = 0.5
    x = np.arange(0 + dx / 2, 1, dx)
    time = np.arange(0, tf, dt)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)

    ims = []
    sol_exa = np.zeros(((solver.mesh.num_cells_x, solver.mesh.num_cells_y, len(time))))
    x = solver.mesh.cells.centroids[..., 1, 0]
    sigma = 125.0 * 1000.0 / (33.0 ** 2)
    c = 0.5
    for i in range(solver.mesh.num_cells_x):
        for j in range(solver.mesh.num_cells_y):
            for k in range(len(time)):
                sol_exa[i, j, k] = math.exp(
                    -sigma * math.fabs(x[i, j] - time[k] - c) ** 2
                )
    erreur = np.zeros(len(time))
    norme = np.zeros(len(time))
    for i, t in enumerate(time):
        x = solver.mesh.cells.centroids[..., 1, 0]
        x = x.reshape(x.size)
        u = solver.mesh.cells.values[..., 1, 0]
        erreur[i] = np.linalg.norm(u - sol_exa[:, :, i])
        norme[i] = np.linalg.norm(sol_exa[:, :, i])

        u = u.reshape(u.size)
        if plot:
            (im1,) = ax1.plot(x, u, "ro-")
            ims.append([im1])

        # Check same solution with 1D-only
        # assert np.sum(err < tol) == len(x)
        solver.step(dt)

    norme_finale = np.linalg.norm(norme, np.inf)
    erreur_finale = np.linalg.norm(erreur, np.inf) / norme_finale
    erreur_finale = erreur[-1] / norme[-1]

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
