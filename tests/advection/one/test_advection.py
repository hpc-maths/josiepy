# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.animation import ArtistAnimation


import josie.general.schemes.time as time_schemes

from josie.general.schemes.space.godunov import Godunov
from josie.general.schemes.space.muscl import MUSCL, MUSCL_Hancock
from josie.general.schemes.space.limiters import No_Limiter
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.solver import Solver

from josie.advection.schemes import Upwind
from josie.advection.problem import AdvectionProblem
from josie.advection.state import Q

# Advection velocity in x-direction
V = np.array([1.0, 0.0])


@pytest.fixture(
    params=sorted(
        [member[1] for member in inspect.getmembers(time_schemes, inspect.isclass)],
        key=lambda c: c.__name__,
    ),
)
def TimeScheme(request):
    yield request.param


@pytest.fixture(
    params=[Godunov, MUSCL, MUSCL_Hancock],
)
def SpaceScheme(request):
    if request.param == Godunov:
        yield request.param

    else:

        class Scheme(request.param, No_Limiter):
            pass

        yield Scheme


@pytest.fixture
def scheme(TimeScheme, SpaceScheme):
    class Scheme(Upwind, TimeScheme, SpaceScheme):
        pass

    yield Scheme


@pytest.fixture
def solver(scheme, boundaries, init):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(40, 1)
    mesh.generate()
    solver = Solver(mesh, Q, scheme(AdvectionProblem(V)))

    def init_fun(cells: MeshCellSet):
        xc = cells.centroids[..., [0]]

        cells.values = init(np.array(xc)).view(Q)

    solver.init(init_fun)

    yield solver


def test_advection(solver, plot, init):
    """Testing against the real 1D solver"""

    nx = solver.mesh.num_cells_x

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ims = []

    # CFL condition
    c = 0.7
    dx = 1 / nx
    dt = c * dx
    T = 0.1

    x = solver.mesh.cells.centroids[..., 0]
    x = x.reshape(x.size)
    Nt = int(np.ceil(T / dt))
    dt = T / Nt

    for t in np.linspace(0, T, Nt + 1):
        u = solver.mesh.cells.values[..., 0]
        u = u.reshape(u.size)

        err = u - init(x - t)

        if plot:
            (im1,) = ax1.plot(x, u, "ro-")
            (im2,) = ax1.plot(x, init(x - t), "ks-")
            ims.append([im1, im2])
            (im_err,) = ax2.plot(x, err, "ks-")
            ims.append([im1, im2, im_err])

        solver.step(dt)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=100)
        plt.show()
