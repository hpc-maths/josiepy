import matplotlib.pyplot as plt
import numpy as np


def test_interpolate(mesh, plot):
    x, y = (mesh._x, mesh._y)

    # Test all the points on the boundary are equal to the points calculated
    # directly using the BoundaryCurves
    xis = np.linspace(0, 1, mesh._num_xi)
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
    if plot:
        mesh.plot()
        plt.show()
