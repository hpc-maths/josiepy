import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.animation import ArtistAnimation

from .adv1d import main as main_1d

from josie.mesh import SimpleCell
from josie.solver.state import StateTemplate
from josie.solver.problem import Problem
from josie.solver.solver import Solver


class Advection(Problem):
    Q = StateTemplate('u')
    # Advection velocity in x-direction
    V = np.array([1, 0])

    @classmethod
    def flux(cls, state_array: np.ndarray) -> np.ndarray:
        # I multiply each element of the given state array by the velocity
        # vector. I obtain an Nx2 array where each row is the flux on each
        # cell
        return cls.V*state_array[:, np.newaxis]


def upwind(values: np.ndarray, neigh_values: np.ndarray,
           normals: np.ndarray, surfaces: np.ndarray):

    F = np.zeros_like(values)

    flux = Advection.flux

    # I do a dot product of each normal in `norm` by the advection velocity
    # Equivalent to: un = np.sum(Advection.V*(normals), axis=-1)
    un = np.einsum('ijk,jk->ij', normals, Advection.V[np.newaxis, :])

    # Check where un > 0
    un_positive = np.all(un > 0)

    # Here the einsum is equivalent to a dot product element by element
    # of flux and norm
    if un_positive:
        F = F + \
            np.einsum('ijkl,ijl->ijk', flux(values), normals)
    else:
        F = F + \
            np.einsum('ijkl,ijl->ijk', flux(neigh_values), normals)

    FS = np.einsum('ijk,ik->ij', F, surfaces)

    return FS[:, np.newaxis, :]


@pytest.fixture
def solver(mesh, init_fun):
    mesh.interpolate(100, 1)
    mesh.generate(SimpleCell)
    solver = Solver(mesh, Advection.Q)
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
        x = solver.mesh.centroids[:, :, 0]
        x = x.reshape(x.size)
        u = solver.values[:, :, 0]
        u = u.reshape(u.size)

        err = u - solution[i, :]

        if plot:
            im1, = ax1.plot(x, u, 'ro-')
            im2, = ax1.plot(x_1d, solution[i, :], 'ks-')
            ims.append([im1, im2])
            im_err, = ax2.plot(x_1d, err, 'ks-')
            ims.append([im1, im2, im_err])

        # Check same solution with 1D-only
        # assert np.sum(err < tol) == len(x)

        solver.step(dt, upwind)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
