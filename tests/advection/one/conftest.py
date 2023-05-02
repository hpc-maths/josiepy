# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from josie.bc import make_periodic, Direction
from josie.boundary import Line
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell

import numpy as np


@pytest.fixture
def boundaries():
    """1D problem along x"""
    left = Line([0, 0], [0, 1])
    bottom = Line([0, 0], [1, 0])
    right = Line([1, 0], [1, 1])
    top = Line([0, 1], [1, 1])

    left, right = make_periodic(left, right, Direction.X)
    top.bc = None
    bottom.bc = None

    yield (left, bottom, right, top)


@pytest.fixture()
def mesh(boundaries):
    left, bottom, right, top = boundaries

    mesh = Mesh(left, bottom, right, top, SimpleCell)
    mesh.interpolate(40, 1)
    mesh.generate()

    yield mesh


@pytest.fixture
def init():
    def init(x: np.ndarray):
        """Init function."""

        a = 0.05
        b = 0.85

        return np.where(
            (x > a) * (x < b),
            np.exp(-1 / (x - a) ** 2)
            * np.exp(-1 / (x - b) ** 2)
            / (
                np.exp(-1 / ((a + b) / 2 - a) ** 2)
                * np.exp(-1 / ((a + b) / 2 - b) ** 2)
            ),
            0,
        )

    yield init
