# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" This module implements the different curve parametrization to describe the
boundaries of a domain
"""
from __future__ import annotations

import abc
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from typing import TYPE_CHECKING

from josie.data import NoAliasIntEnum
from josie.geometry import MeshIndex, PointType
from josie.math import map01to


if TYPE_CHECKING:
    from josie.bc import BoundaryCondition
    from josie.mesh.cellset import MeshCellSet


class BoundarySide(NoAliasIntEnum):
    LEFT = 0
    RIGHT = -1
    TOP = -1
    BOTTOM = 0


@dataclass
class Boundary:
    """A simple :class:`dataclass` coupling a :class:`~.BoundaryCurve` with the
    indices of the cells within the :attr:`Mesh.centroids` data structure that
    are part of that boundary :attr:`cells_idx`, the side of the boundary
    (class:`BoundarySide`), and the corresponding indices of the ghost cells
    within the :attr:`Mesh._centroids`


    Attributes
    ----------
    side
        The :class:`BoundarySide` which the :class:`Boundary` is member of

    boundary_curve
        The :class:`~.BoundaryCurve`

    cells_idx
        The cell indices. It's a tuple containing
        :data:`~josie._dim.MAX_DIMENSIONALITY` :class:`MeshIndex` elements.
        Each element indexes the structured :class:`Mesh` on one axis to
        provide the cells that are part of the :class:`BoundaryCurve`.

    ghost_cells_idx
        The ghost cell indices. It's a tuple containing
        :data:`~josie._dim.MAX_DIMENSIONALITY` :class:`MeshIndex` elements.
        Each element indexes the structured :class:`Mesh` on one axis to
        provide the cells that are part of the :class:`BoundaryCurve`.

    Example
    -------
    For example the left boundary of a 2D structured mesh will be given by a
    tuple (0, None). That means that if we consider the :class:`Solver`, we can
    access the values of the fields associated to the boundary cells by:

    .. code-block:: python

        solver.values[0, :]
    """

    # TODO: Generalize for 3D and unstructured

    side: BoundarySide
    curve: BoundaryCurve
    cells_idx: MeshIndex
    ghost_cells_idx: MeshIndex

    def init_bc(self, cells: MeshCellSet):
        """Proxy method to :meth:`BoundaryCurve.init`"""

        self.curve.bc.init(cells, self)

    def apply_bc(self, cells: MeshCellSet, t: float):
        """Proxy method to :meth:`BoundaryCurve.bc`"""

        self.curve.bc(cells, self, t)


class BoundaryCurve(metaclass=abc.ABCMeta):
    r"""A class representing a :class:`BoundaryCurve`. A
    :class:`BoundaryCurve` is parametrized with a single parameter. It
    implements a :meth:`__call__` method that returns the :math:`(x,y)` values
    of the curve for a given :math:`\xi` parameter value.
    """

    @property
    def bc(self):
        return self._bc

    @bc.setter
    def bc(self, bc: BoundaryCondition):
        self._bc = bc

    @abc.abstractmethod
    def __call__(self, xi):
        r"""The effective parametrization of the BoundaryCurve. Assume ``xi``
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
        """This method actually plots the BoundaryCurve

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
    """A line between two points

    Parameters
    ---------
    p1
        Starting point of the line
    p2
        Ending point of the line
    """

    def __init__(self, p1: PointType, p2: PointType):
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
