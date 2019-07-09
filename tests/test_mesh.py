import matplotlib.pyplot as plt
import numpy as np
import pytest

from josie.geom import Line, CircleArc
from josie.mesh import Mesh


@pytest.fixture
def boundaries():
    left = Line([0, 0], [0, 1])
    bottom = CircleArc([0, 0], [1, 0], [0.5, 0.5])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    yield (left, bottom, right, top)


def test_interpolate(boundaries, plot):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top)
    x, y = mesh.interpolate(20, 20)

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
        plt.figure()
        plt.plot(x, y, 'k.')
        left.plot()
        bottom.plot()
        right.plot()
        top.plot()
        plt.axis('equal')
        plt.show()
