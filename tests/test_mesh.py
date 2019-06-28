import matplotlib.pyplot as plt
import numpy as np

from josie.geom import Line, CircleArc
from josie.mesh import Mesh


def test_mesh(plot):
    left = Line([0, 0], [0, 1])
    bottom = CircleArc([0, 0], [1, 0], [0.5, 0.5])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    mesh = Mesh(left, bottom, right, top)
    x, y = mesh.generate(20, 20)

    # Test all the points on the boundary are equal to the points calculated
    # directly using the BoundaryCurves
    xis = np.linspace(0, 1, 20)
    xl, yl = left(xis)
    xr, yr = right(xis)
    xt, yt = top(xis)
    xb, yb = bottom(xis)

    assert np.allclose(x[0, :], xl) and np.allclose(y[0, :], yl)
    assert np.allclose(x[-1, :], xr) and np.allclose(y[0, :], yr)
    assert np.allclose(x[:, 0], xb) and np.allclose(y[:, 0], yb)
    assert np.allclose(x[:, -1], xt) and np.allclose(y[:, -1], yt)

    if plot:
        plt.plot(x, y, 'k.')
        left.plot()
        bottom.plot()
        right.plot()
        top.plot()
        plt.axis('equal')
        plt.show()
