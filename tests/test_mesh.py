import matplotlib.pyplot as plt
import numpy as np
import pytest

from josie.geom import Line, CircleArc
from josie.mesh import Mesh


@pytest.fixture
def boundaries():
    left = Line([0, 0], [0, 1])
    bottom = CircleArc([0, 0], [1, 0], [0.2, 0.2])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    bc = lambda mesh, cell: 0

    left.bc = bc
    bottom.bc = bc
    right.bc = bc
    top.bc = bc

    yield (left, bottom, right, top)


@pytest.fixture
def mesh(boundaries):
    left, bottom, right, top = boundaries

    yield Mesh(left, bottom, right, top)


def test_interpolate(mesh, plot):
    x, y = mesh.interpolate(20, 20)

    # Test all the points on the boundary are equal to the points calculated
    # directly using the BoundaryCurves
    xis = np.linspace(0, 1, 20)
    xl, yl = mesh.left(xis)
    xr, yr = mesh.right(xis)
    xt, yt = mesh.top(xis)
    xb, yb = mesh.bottom(xis)

    assert np.allclose(x[0, :], xl) and np.allclose(y[0, :], yl)
    assert np.allclose(x[-1, :], xr) and np.allclose(y[0, :], yr)
    assert np.allclose(x[:, 0], xb) and np.allclose(y[:, 0], yb)
    assert np.allclose(x[:, -1], xt) and np.allclose(y[:, -1], yt)

    if plot:
        plt.figure()
        plt.plot(x, y, 'k.')
        mesh.left.plot()
        mesh.bottom.plot()
        mesh.right.plot()
        mesh.top.plot()
        plt.axis('equal')
        plt.show(block=False)


def test_plot(mesh, plot):
    mesh.interpolate(20, 20)
    mesh.generate()
    if plot:
        plt.figure()
        mesh.plot()
        plt.show()
