import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.animation import ArtistAnimation

from .adv1d import main as main_1d

from josie.solver.problem import Problem
from josie.solver.solver import Solver
from josie.solver.state import StateTemplate, State


class Advection(Problem):
    # Scalar advection
    Q = StateTemplate('u')

    # Advection velocity in x-direction
    V = np.array([1, 0])

    @classmethod
    def flux(cls, Q: State) -> np.ndarray:
        return cls.V*Q


def upwind(cell):
    Q = Advection.Q(0)

    for neigh in cell:
        norm = neigh.face.normal
        flux = Advection.flux
        S = neigh.face.surface

        un = Advection.V.dot(norm)

        if un >= 0:
            Q = Q + flux(cell.value).dot(norm)*S
        else:
            Q = Q + flux(neigh.value).dot(norm)*S

    return Q


@pytest.fixture
def solver(mesh, init_fun):
    mesh.interpolate(100, 1)
    mesh.generate()
    solver = Solver(mesh, Advection)
    solver.init(init_fun)

    yield solver


def test_against_real_1D(solver, plot, tol):
    """ Testing against the real 1D solver """

    time, x_1d, solution = main_1d(100, 4, 0.9, plot)
    dt = time[1] - time[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ims = []

    for i, t in enumerate(time):
        x = np.asarray([cell.centroid[0] for cell in
                        solver.mesh.cells.ravel()])
        u = np.asarray([cell.value for cell in solver.mesh.cells.ravel()])
        u = u.flatten()

        err = u - solution[i, :]

        if plot:
            im1, = ax1.plot(x, u, 'ro-')
            im2, = ax1.plot(x_1d, solution[i, :], 'ks-')
            ims.append([im1, im2])
            im_err, = ax2.plot(x_1d, err, 'ks-')
            ims.append([im1, im2, im_err])

        # Check same solution with 1D-only
        assert np.sum(err < tol) == len(x)

        solver.step(dt, upwind)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
