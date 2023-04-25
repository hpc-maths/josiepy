import abc
import numpy as np
import pytest
import math

from josie.bc import make_periodic, Direction, Neumann
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import DGCell

from josie.general.schemes.time.rk import RK, ButcherTableau
from josie.mesh.cellset import MeshCellSet, CellSet
from josie.state import State, StateTemplate
from josie.solver import Solver
from josie.problem import Problem
from josie.pgd.schemes.LF import LF
from josie.pgd.problem import PGDProblem
from josie.pgd.state import PGDState
from josie.pgd.solver import PGDSolver
from josie.pgd.fields import PGDFields
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.animation import ArtistAnimation


from typing import NoReturn, Callable


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
        # Limiter PGD 1D
        self.limiter(step_cells)

        step_cells.update_ghosts(mesh.boundaries, t)

        self.pre_accumulate(step_cells, t)

        for neighs in step_cells.neighbours:
            self.accumulate(step_cells, neighs, t)

        vec = np.einsum(
            "ijk,...jmk->...imk",
            self.K_ref,
            self.problem.F(step_cells),
        )
        dx = mesh._x[1, 0] - mesh._x[0, 0]
        dy = mesh._y[0, 1] - mesh._y[0, 0]
        vec2 = (2.0 / dx) * vec[..., 0] + (2.0 / dy) * vec[..., 1]

        self._fluxes.view(PGDState).set_conservative(
            self._fluxes.view(PGDState).get_conservative() - vec2
        )

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


class SolverDG(PGDSolver):
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


@pytest.fixture
def scheme():
    class Test_scheme(LF, RKDG2):
        pass

    yield Test_scheme()


@pytest.fixture
def solver(scheme):
    """1D problem along x"""
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    dQ = np.zeros(len(PGDState.fields)).view(PGDState)
    left.bc = Neumann(dQ)
    right.bc = Neumann(dQ)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top, DGCell)
    mesh.interpolate(50, 1)
    mesh.generate()

    solver = SolverDG(mesh, scheme)

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., 0]
        for i in range(mesh.num_cells_x):
            for j in range(mesh.num_cells_y):
                for k in range(mesh.cell_type.num_dofs):
                    # Test case Bouchut vacuum
                    # cells.values[i, j, k, PGDFields.rho] = 0.5
                    # if xc[i, j, k] < 0.5:
                    #     cells.values[i, j, k, PGDFields.rhoU] = (
                    #         -0.4 * cells.values[i, j, k, PGDFields.rho]
                    #     )
                    #     cells.values[i, j, k, PGDFields.U] = -0.4
                    # elif xc[i, j, k] >= 0.5 and xc[i, j, k] < 1.0:
                    #     cells.values[i, j, k, PGDFields.rhoU] = (
                    #         0.4 * cells.values[i, j, k, PGDFields.rho]
                    #     )
                    #     cells.values[i, j, k, PGDFields.U] = 0.4
                    # elif xc[i, j, k] >= 1.0 and xc[i, j, k] < 1.8:
                    #     cells.values[i, j, k, PGDFields.rhoU] = (
                    #         1.4 - xc[i, j, k]
                    #     ) * cells.values[i, j, k, PGDFields.rho]
                    #     cells.values[i, j, k, PGDFields.U] = 1.4 - xc[i, j, k]
                    # else:
                    #     cells.values[i, j, k, PGDFields.rhoU] = (
                    #         -0.4 * cells.values[i, j, k, PGDFields.rho]
                    #     )
                    #     cells.values[i, j, k, PGDFields.U] = -0.4

                    # Test case delta shock
                    cells.values[i, j, k, PGDFields.rho] = math.pow(
                        math.sin(2 * math.pi * xc[i, j, k]), 4
                    )
                    # if xc[i, j, k] > 0.5:
                    #     cells.values[i, j, k, PGDFields.rhoU] = (
                    #         -1.0 * cells.values[i, j, k, PGDFields.rho]
                    #     )
                    #     cells.values[i, j, k, PGDFields.U] = -1.0
                    # else:
                    #     cells.values[i, j, k, PGDFields.rhoU] = (
                    #         1.0 * cells.values[i, j, k, PGDFields.rho]
                    #     )
                    #     cells.values[i, j, k, PGDFields.U] = 1.0
                    # Test case delta shock 2
                    if xc[i, j, k] > 0.5:
                        cells.values[i, j, k, PGDFields.rhoU] = (
                            -xc[i, j, k]
                        ) * cells.values[i, j, k, PGDFields.rho]
                        cells.values[i, j, k, PGDFields.U] = -xc[i, j, k]
                    else:
                        cells.values[i, j, k, PGDFields.rhoU] = (
                            -xc[i, j, k] + 1.0
                        ) * cells.values[i, j, k, PGDFields.rho]
                        cells.values[i, j, k, PGDFields.U] = -xc[i, j, k] + 1.0
                    # Test case transport gaussienne
                    # if xc[i, j, k] > 0.25 and xc[i, j, k] < 0.75:
                    #     cells.values[i, j, k, PGDFields.rho] = math.pow(
                    #         math.cos(math.pi * (2 * xc[i, j, k] - 1)), 4
                    #     )
                    # else:
                    #     cells.values[i, j, k, PGDFields.rho] = 1e-10
                    # Test case transport creneau
                    # if xc[i, j, k] > 0.45:
                    #     cells.values[i, j, k, PGDFields.rho] = 1.0
                    # else:
                    #     cells.values[i, j, k, PGDFields.rho] = 1e-10
                    # # Test case etat constant
                    # # cells.values[i, j, k, PGDFields.rho] = 1.0
                    # cells.values[i, j, k, PGDFields.rhoU] = (
                    #    1.0 * cells.values[i, j, k, PGDFields.rho]
                    # )
                    cells.values[i, j, k, PGDFields.rhoV] = 0.0
                    # cells.values[i, j, k, PGDFields.U] = 1.0
                    cells.values[i, j, k, PGDFields.V] = 0.0

    solver.init(init_fun)
    solver.scheme.init_limiter(solver.mesh.cells)

    yield solver


def test_against_real_1D(solver, plot):
    """Testing against the real 1D solver"""

    rLGLmin = 2.0
    cfl = 0.1
    dx = solver.mesh._x[1, 0] - solver.mesh._x[0, 0]
    tf = 0.5

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ims = []

    t = 0.0
    while t < tf:
        maxvel = np.amax(np.abs(solver.mesh.cells.values[..., PGDFields.U]))
        dt = cfl * rLGLmin * dx / maxvel
        x = solver.mesh.cells.centroids[..., 1, 0]
        rho = solver.mesh.cells.values[..., 1, PGDFields.rho]
        Ux = solver.mesh.cells.values[..., 1, PGDFields.U]
        if plot:
            (im1,) = ax1.plot(x, rho, "ro-")
            (im2,) = ax2.plot(x, Ux, "ro-")
            ims.append([im1, im2])

        solver.step(dt)
        t += dt

    if plot:
        ani = ArtistAnimation(fig, ims)
        ani.save("PGD_1D_d-choc2_50x1.mp4", writer="ffmpeg")
        plt.show()
