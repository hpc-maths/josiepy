# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Iterator

from .boundary import BoundaryCurve, Line


@dataclass
class BoundarySet:
    left: BoundaryCurve
    bottom: BoundaryCurve
    right: BoundaryCurve
    top: BoundaryCurve

    def __iter__(self) -> Iterator[BoundaryCurve]:
        return iter([self.left, self.bottom, self.right, self.top])


class BoxMesh(BoundarySet):
    """A convenience class to create a rectangular mesh provided its
    dimensions

    Parameters
    ----------
    length
        The length of the box

    height
        The height of the box
    """

    def __init__(
        self,
        length: float,
        height: float,
    ):
        left = Line([0, 0], [0, height])
        bottom = Line([0, 0], [length, 0])
        right = Line([length, 0], [length, height])
        top = Line([0, height], [length, height])

        super().__init__(left, bottom, right, top)


class Cube(BoxMesh):
    """A :class:`BoxMesh` with equal lenght and height"""

    def __init__(
        self,
        side_length: float,
    ):

        super().__init__(side_length, side_length)


class UnitCube(Cube):
    """:class:`Cube` with length equal to 1"""

    def __init__(
        self,
    ):

        super().__init__(1.0)
