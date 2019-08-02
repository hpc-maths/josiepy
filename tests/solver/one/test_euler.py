import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import ArtistAnimation

from josie.bc import Dirichlet
from josie.geom import Line
from josie.mesh import Mesh
from josie.solver import Problem, Solver, State, StateTemplate


class EOS:
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def rhoe_from_rho_p(self, rho, p):
        """ This returns the internal energy multiplied by the density"""

        return p/(self.gamma - 1)

    def p_from_rho_e(self, rho, e):
        """ This returns the pressure from internal energy and density """
        return (self.gamma - 1)*rho*e

    def sound_velocity(self, rho, p):
        """ This returns the sound velocity from density and pressure"""

        return np.sqrt(self.gamma*p/rho)


class Euler(Problem):
    Q = StateTemplate("rho", "rhoU", "rhoV", "rhoE", "rhoe", "U", "V", "p",
                      "c")
    eos = EOS()

    @classmethod
    def flux(cls, Q: State) -> np.ndarray:

        return np.array([
            [Q.rhoU, Q.rhoV],
            [Q.rhoU*Q.U + Q.p, Q.rhoU*Q.V],
            [Q.rhoV*Q.U, Q.rhoV*Q.V + Q.p],
            [(Q.rhoE + Q.p)*Q.U, (Q.rhoE + Q.p)*Q.V]
        ])


class EulerSolver(Solver):
    # I redefine the step since we need to update the auxiliary variables

    def post_step(self):
        for cell in self.mesh.cells.ravel():
            Q = cell.value
            rho = Q.rho
            U = Q.rhoU/rho
            V = Q.rhoV/rho
            rhoe = Q.rhoE - 0.5*rho*(U**2 + V**2)
            e = rhoe/rho
            p = Euler.eos.p_from_rho_e(rho, e)
            c = Euler.eos.sound_velocity(rho, p)
            cell.value[4:] = np.array([rhoe, U, V, p, c])


def test_toro_1(plot):
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    # BC
    rhoL = 1.0
    uL = 0.0
    vL = 0.0
    pL = 1.0
    rhoeL = Euler.eos.rhoe_from_rho_p(rhoL, pL)
    EL = rhoeL/rhoL + 0.5*(uL**2 + vL**2)
    cL = Euler.eos.sound_velocity(rhoL, pL)

    rhoR = 0.125
    uR = 0.0
    vR = 0.0
    pR = 0.1
    rhoeR = Euler.eos.rhoe_from_rho_p(rhoR, pR)
    ER = rhoeR/rhoR + 0.5*(uR**2 + vR**2)
    cR = Euler.eos.sound_velocity(rhoR, pR)

    Q_left = Euler.Q(rhoL, rhoL*uL, rhoL*vL, rhoL*EL, rhoeL, uL, vL, pL, cL)
    Q_right = Euler.Q(rhoR, rhoR*uR, rhoR*vR, rhoR*ER, rhoeR, uR, vR, pR, cR)

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

    solver = EulerSolver(mesh, Euler)
    solver.init(init_fun)

    x = [cell.centroid[0] for cell in solver.mesh.cells.ravel()]
    Q = [cell.value for cell in solver.mesh.cells.ravel()]

    def rusanov(cell):
        Q = Euler.Q(0, 0, 0, 0, 0, 0, 0, 0, 0)
        # First four variables of the total state are the conservative
        # variables (rho, rhoU, rhoV, rhoE)
        Q_cons = Q[:4]

        Q_cell = cell.value
        Q_cell_cons = Q_cell[:4]

        for neigh in cell:
            # Geometry
            norm = neigh.face.normal
            S = neigh.face.surface

            Q_neigh = neigh.value
            Q_neigh_cons = Q_neigh[:4]

            sigma = np.max((
                np.abs(Q_cell.U) + Q_cell.c,
                np.abs(Q_neigh.U) + Q_neigh.c
            ))

            flux = Euler.flux

            F = 0.5*(flux(Q_cell) + flux(Q_neigh)).dot(norm) - \
                0.5*sigma*(Q_neigh_cons - Q_cell_cons)

            Q_cons = Q_cons + F*S

        Q[:4] = Q_cons

        return Q

    dt = 8E-4
    time = np.arange(0, 0.25, dt)

    if plot:
        ims = []
        fig = plt.figure()

    for t in time:
        print(t)
        x = np.asarray([cell.centroid[0] for cell in
                        solver.mesh.cells.ravel()])
        Q = np.asarray([cell.value for cell in solver.mesh.cells.ravel()])

        if plot:
            im1, = plt.plot(x, Q[:, 0], 'k-')
            ims.append([im1])

        solver.step(dt, rusanov)

    if plot:
        _ = ArtistAnimation(fig, ims, interval=50)
        plt.show()
