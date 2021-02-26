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
    xb, yb = mesh.bottom.curve(xis)

    assert np.allclose(x[0, :], xl) and np.allclose(y[0, :], yl)
    assert np.allclose(x[-1, :], xr) and np.allclose(y[0, :], yr)
    assert np.allclose(x[:, 0], xb) and np.allclose(y[:, 0], yb)
    assert np.allclose(x[:, -1], xt) and np.allclose(y[:, -1], yt)

    plt.figure()
    plt.plot(x, y, "k.")
    mesh.left.curve.plot()
    mesh.bottom.curve.plot()
    mesh.right.curve.plot()
    mesh.top.curve.plot()
    plt.axis("equal")
    plt.show(block=False)


def test_ghost_centroids(mesh, tol):
    # Test that the ghosts centroids have distance one from the corresponding
    # boundary cells

    for boundary in mesh.boundaries:
        boundary_idx = boundary.cells_idx
        ghost_idx = boundary.ghost_cells_idx

        boundary_centroids = mesh.cells._centroids[
            boundary_idx[0], boundary_idx[1]
        ]

        # Compute the ghost cells centroids
        ghost_centroids = mesh.cells._centroids[ghost_idx[0], ghost_idx[1]]

        # The distance (lenght of the relative vector between boundary cells
        # and related ghost cell) must be min_length
        distances = ghost_centroids - boundary_centroids
        assert (
            np.mean(np.linalg.norm(distances, axis=-1)) - mesh.cells.min_length
            < tol
        )


def test_write(tmp_path, mesh):
    mesh.generate()
    file = tmp_path / "test.xdmf"
    mesh.write(file.as_posix())


def test_plot(mesh, plot):
    mesh.plot()
