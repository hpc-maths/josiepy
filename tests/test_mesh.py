import matplotlib.pyplot as plt
import numpy as np


def test_interpolate(mesh, plot):
    x, y = (mesh._x, mesh._y)

    # Test all the points on the boundary are equal to the points calculated
    # directly using the BoundaryCurves
    xis = np.linspace(0, 1, mesh._num_xi)
    xl, yl = mesh.left.curve(xis)
    xr, yr = mesh.right.curve(xis)
    xt, yt = mesh.top.curve(xis)
    xb, yb = mesh.btm.curve(xis)

    assert np.allclose(x[0, :], xl) and np.allclose(y[0, :], yl)
    assert np.allclose(x[-1, :], xr) and np.allclose(y[0, :], yr)
    assert np.allclose(x[:, 0], xb) and np.allclose(y[:, 0], yb)
    assert np.allclose(x[:, -1], xt) and np.allclose(y[:, -1], yt)

    plt.figure()
    plt.plot(x, y, "k.")
    mesh.left.curve.plot()
    mesh.btm.curve.plot()
    mesh.right.curve.plot()
    mesh.top.curve.plot()
    plt.axis("equal")
    plt.show(block=False)


def test_write(tmp_path, mesh):
    mesh.generate()
    file = tmp_path / "test.xdmf"
    mesh.write(file.as_posix())


def test_plot(mesh, plot):
    mesh.plot()
