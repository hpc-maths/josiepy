import numpy as np

from josie.mesh.cell import Cell


def test_square_centroid(tol):
    c = Cell(
        (-1, 1),
        (-1, -1),
        (1, -1),
        (1, 1),
        0,
        0
    )

    assert np.array_equal(c.centroid, np.array((0, 0)))
    assert c.volume - 4 < tol


def test_non_equilateral_cell(tol):
    p1 = np.array((-1, 1))
    p2 = np.array((-1, -1))
    p3 = np.array((1, 0))
    p4 = np.array((0, 2))

    c = Cell(p1, p2, p3, p4, 0, 0)

    centroid = (p1 + p2 + p3 + p4)/4
    area = np.linalg.norm(
        np.cross(
            p2 - p1,
            p3 - p2
        )
    )/2 + np.linalg.norm(
        np.cross(
            p4 - p3,
            p1 - p4
        )
    )/2

    assert np.array_equal(c.centroid, centroid)
    assert c.volume == area
