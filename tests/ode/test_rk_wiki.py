""" Testing
this:https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods#Use
"""
import numpy as np

from josie.ode import OdeSolver
from josie.solver.state import StateTemplate
from josie.general.schemes.time import RK2

Q = StateTemplate("y")

solution = [1, 1.066869388, 1.141332181, 1.227417567, 1.335079087]


def test_rk_wiki(mesh, plot):
    """Testing against the analytical solution of an harmonic oscillator
    without damping"""

    dt = 0.025
    Q0 = Q(y=1)

    solver = OdeSolver(Q0, dt, RK2, lambda y, t: np.tan(y) + 1)

    # Asserting the value of the state against hardcoded exact values
    # on Wikipedia. Precision is 9 decimal digits
    for step in solution:
        assert np.abs(solver.mesh.cells.values.item() - step) < 1e-9
        solver.step(dt)
