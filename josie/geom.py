# josiepy
# Copyright © 2019 Ruben Di Battista
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

""" This module implements the different curve parametrization to describe the
boundaries of a domain
"""


import abc

import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING
from josie.math import map01to

if TYPE_CHECKING:
    from josie.bc import BoundaryCondition  # pragma: no cover


class BoundaryCurve(metaclass=abc.ABCMeta):
    r""" A class representing a :class:`BoundaryCurve`. A
    :class:`BoundaryCurve` is parametrized with a single parameter. It
    implements a :meth:`__call__` method that returns the :math:`(x,y)` values
    of the curve for a given :math:`\xi` parameter value.
    """

    @property
    def bc(self):
        return self._bc

    @bc.setter
    def bc(self, bc: "BoundaryCondition"):
        self._bc = bc

    @abc.abstractmethod
    def __call__(self, xi):
        r""" The effective parametrization of the BoundaryCurve. Assume ``xi``
        (:math:`\xi`) to range into :math:`[0, 1]`

        Args:
            xi: The parameter the curve is parametrized with.
                Ranges from 0 to 1

        Returns:
            xy: | A tuple containing the x and y coordinates of the
                :class:`BoundaryCurve` computed for the :math:`\xi` values
        """

        raise NotImplementedError  # pragma: no cover

    def plot(self, resolution=50):
        """ This method actually plots the BoundaryCurve

        This method currently plots stuff using matplotlib. It generates the
        list of points to plot with a default `resolution`.

        Args:
            resolution: The number of points to plot the curve with
                [default: 50].
        """

        xi = np.linspace(0, 1, resolution)

        X, Y = self(xi)

        plt.plot(X, Y, "o-")


class Line(BoundaryCurve):
    """ A line between two points

    Parameters:
        p1: Starting point of the line
        p2: Ending point of the line
    """

    def __init__(self, p1, p2):
        self._p1 = p1
        self._p2 = p2

    def __call__(self, xi):
        x = (1 - xi) * self._p1[0] + xi * self._p2[0]
        y = (1 - xi) * self._p1[1] + xi * self._p2[1]

        return (x, y)


class CircleArc(BoundaryCurve):
    """A class representing a circular arc from three given points on the
    arc


    Parameters:
        p1: Starting point of the arc
        p2: Ending point of the arc
        p3: Another point on the arc
    """

    def __init__(self, p1, p2, p3):
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3

        # Find the circle passing by the three points
        A = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])

        b = -np.array(
            [
                p1[0] ** 2 + p1[1] ** 2,
                p2[0] ** 2 + p2[1] ** 2,
                p3[0] ** 2 + p3[1] ** 2,
            ]
        )

        X = np.linalg.solve(A, b)

        # Center coordinates
        self._c = np.array([-X[0] / 2, -X[1] / 2])

        self._r1 = self._p1 - self._c
        self._r2 = self._p2 - self._c
        self._r = np.linalg.norm(self._r1)

        # Starting angles
        cos1 = self._r1[0] / self._r
        cos2 = self._r2[0] / self._r

        sin1 = self._r1[1] / self._r
        sin2 = self._r2[1] / self._r

        th1 = np.arctan2(sin1, cos1)
        th2 = np.arctan2(sin2, cos2)

        # Need to renormalize between [0, 2pi]
        th1 = (th1 + 2 * np.pi) % (2 * np.pi)
        th2 = (th2 + 2 * np.pi) % (2 * np.pi)

        self._th1 = th1
        self._th2 = th2

    def __call__(self, xi):

        # Remap to the correct angle range
        xi = map01to(xi, self._th1, self._th2)

        return (
            self._c[0] + self._r * np.cos(xi),
            self._c[1] + self._r * np.sin(xi),
        )
