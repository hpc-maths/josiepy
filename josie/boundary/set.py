# josiepy
# Copyright © 2020 Ruben Di Battista
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Ruben Di Battista ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Ruben Di Battista BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Ruben Di Battista.
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
    """ A :class:`BoxMesh` with equal lenght and height """

    def __init__(
        self,
        side_length: float,
    ):

        super().__init__(side_length, side_length)


class UnitCube(Cube):
    """ :class:`Cube` with length equal to 1 """

    def __init__(
        self,
    ):

        super().__init__(1.0)
