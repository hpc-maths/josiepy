# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cell import SimpleCell


def test_square_centroid(tol):
    pts = (
        (-1, 1),
        (-1, -1),
        (1, -1),
        (1, 1),
    )

    centroid = SimpleCell.centroid(pts[0], pts[1], pts[2], pts[3])
    volume = SimpleCell.volume(pts[0], pts[1], pts[2], pts[3])

    assert np.array_equal(centroid, np.array((0, 0)))
    assert volume - 4 < tol


def test_non_equilateral_cell(tol):
    p1 = np.array((-1, 1))
    p2 = np.array((-1, -1))
    p3 = np.array((1, 0))
    p4 = np.array((0, 2))

    cell_centroid = SimpleCell.centroid(p1, p2, p3, p4)
    cell_volume = SimpleCell.volume(p1, p2, p3, p4)

    centroid = (p1 + p2 + p3 + p4) / 4
    area = (
        np.linalg.norm(np.cross(p2 - p1, p3 - p2)) / 2
        + np.linalg.norm(np.cross(p4 - p3, p1 - p4)) / 2
    )

    assert np.array_equal(cell_centroid, centroid)
    assert cell_volume == area
