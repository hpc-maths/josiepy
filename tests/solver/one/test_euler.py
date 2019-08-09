import matplotlib.pyplot as plt
import numpy as np
import pytest

from matplotlib.animation import ArtistAnimation

from josie.bc import Dirichlet
from josie.geom import Line
from josie.mesh import Mesh
from josie.solver.euler import rusanov, Q, EulerSolver, PerfectGas

riemann_states = [
    {
        'rhoL': 1.0,
        'uL': 0.0,
        'vL': 0,
        'pL': 1.0,
        'rhoR': 0.125,
        'uR': 0,
        'vR': 0,
        'pR': 0.1,
        'dt': 8E-4,
    },
    {
        'rhoL': 1.0,
        'uL': -2,
        'vL': 0,
        'pL': 0.4,
        'rhoR': 1.0,
        'uR': 2.0,
        'vR': 0,
        'pR': 0.4,
        'dt': 1E-4,
    },
    {
        'rhoL': 1.0,
        'uL': 0,
        'vL': 0,
        'pL': 1000,
        'rhoR': 1.0,
        'uR': 0,
        'vR': 0,
        'pR': 0.01,
        'dt': 1E-5,
    },
    {
        'rhoL': 5.99924,
        'uL': 19.5975,
        'vL': 0,
        'pL': 460.894,
        'rhoR': 5.9924,
        'uR': -6.19633,
        'vR': 0,
        'pR': 46.0950,
        'dt': 1E-5,
    },
    {
        'rhoL': 1.0,
        'uL': -19.59745,
        'vL': 0,
        'pL': 1000,
        'rhoR': 1.0,
        'uR': -19.59745,
        'vR': 0,
        'pR': 0.01,
        'dt': 1E-5,
    }
]


@pytest.mark.parametrize("riemann_problem", riemann_states)
def test_toro_1(riemann_problem, plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    eos = PerfectGas(gamma=1.4)

    # BC
    rhoL = riemann_problem['rhoL']
    uL = riemann_problem['uL']
    vL = riemann_problem['vL']
    pL = riemann_problem['pL']
    rhoeL = eos.rhoe_from_rho_p(rhoL, pL)
    EL = rhoeL/rhoL + 0.5*(uL**2 + vL**2)
    cL = eos.sound_velocity(rhoL, pL)

    rhoR = riemann_problem['rhoR']
    uR = riemann_problem['uR']
    vR = riemann_problem['vR']
    pR = riemann_problem['pR']
    rhoeR = eos.rhoe_from_rho_p(rhoR, pR)
    ER = rhoeR/rhoR + 0.5*(uR**2 + vR**2)
    cR = eos.sound_velocity(rhoR, pR)

    Q_left = Q(rhoL, rhoL*uL, rhoL*vL, rhoL*EL, rhoeL, uL, vL, pL, cL)
    Q_right = Q(rhoR, rhoR*uR, rhoR*vR, rhoR*ER, rhoeR, uR, vR, pR, cR)

    left.bc = Dirichlet(Q_left)
    right.bc = Dirichlet(Q_right)
    top.bc = None
    bottom.bc = None

    mesh = Mesh(left, bottom, right, top)
    mesh.interpolate(500, 1)
    mesh.generate()

    def init_fun(cell):
        xc, _ = cell.centroid

        if xc > 0.5:
            return Q_right
        else:
            return Q_left

    solver = EulerSolver(mesh, eos)
    solver.init(init_fun)

    dt = riemann_problem['dt']
    time = np.arange(0, 0.25, dt)

    if plot:
        ims = []
        fig = plt.figure()
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

    for t in time:
        print(t)
        x = np.asarray([cell.centroid[0] for cell in
                        solver.mesh.cells.ravel()])
        Q_result = np.asarray([cell.value for cell in
                               solver.mesh.cells.ravel()])

        if plot:
            im1, = ax1.plot(x, Q_result[:, 0], 'k-')
            ax1.set_xlabel('x')
            ax1.set_ylabel(r'$\rho$')

            im2, = ax2.plot(x, Q_result[:, 5], 'k-')
            ax2.set_xlabel('x')
            ax2.set_ylabel('U')

            im3, = ax3.plot(x, Q_result[:, 7], 'k-')
            ax3.set_xlabel('x')
            ax3.set_ylabel('p')

            ims.append([im1, im2, im3])

        solver.step(dt, rusanov)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
        plt.close()
