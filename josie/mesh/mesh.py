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

""" This module contains the primitives related to the mesh generation """

import meshio
import numpy as np
import os

from dataclasses import dataclass
from meshio import Mesh as MeshIO
from typing import Iterable, Tuple, Type, TYPE_CHECKING, Union

from josie.exceptions import InvalidMesh
from josie.geom import BoundaryCurve
from josie.plot import DefaultBackend
from josie.plot.backend import PlotBackend

from .cell import Cell

if TYPE_CHECKING:
    # This is a trick to enable mypy to evaluate the Enum as a standard
    # library Enum for type checking but we use `aenum` in the running code
    from enum import IntEnum  # pragma: no cover

    NoAlias = object()  # pragma: no cover
else:
    from aenum import IntEnum, NoAlias

MeshIndex = Union[int, slice]


@dataclass
class Boundary:
    """ A simple :class:`dataclass` coupling a :class:`~.BoundaryCurve` with
    the indices of the cells that are part of that boundary

    Attributes
    ----------
    boundary_curve
        The :class:`~.BoundaryCurve`

    idx
        The cell indices
    """

    curve: BoundaryCurve
    cells_idx: Tuple[MeshIndex, ...]


class _BoundarySide(IntEnum, settings=NoAlias):
    LEFT = 0
    RIGHT = -1
    TOP = -1
    BOTTOM = 0


class Mesh:
    r""" This class handles the mesh generation over a domain.

    Parameters
    ----------
    left
        The left :class:`BoundaryCurve`

    bottom
        The bottom :class:`BoundaryCurve`

    right
        The right :class:`BoundaryCurve`

    top
        The right :class:`BoundaryCurve`

    cell_type
        A :class:`Cell` class that implements a
        :func:`~Cell.create_connectivity`

    Attributes
    ----------
    left
        The left :class:`Boundary`

    btm
        The bottom :class:`Boundary`

    right
        The right :class:`Boundary`

    top
        The right :class:`Boundary`

    boundaries
        An iterable over the boundaries available based on the
        :attr:`dimensionality`

    oneD: bool
        A flag to indicate if the mesh is 1D or not

    cell_type
        A :class:`Cell` class that implements a
        :func:`~Cell.create_connectivity`

    num_cells_x
        The number of cells in the :math:`x`-direction

    num_cells_y
        The number of cells in the :math:`y`-direction

    centroids
        An array containing the centroid of the cells. It has the dimensions of
        :math:`N_x \times N_y`

    volumes
        An array containing the volumes of the cells. It has the dimensions of
        :math:`N_x \times N_y`

    surfaces
        An array containing the surfaces of the cells. It has the dimensions of
        :math:`N_x \times N_y \times N_\text{points}` where
        N_\text{points} depends on the :class:`Cell` type provided

    points
        An array containing the points that constitute a cell. It has the
        dimensions of :math:`N_x \times N_y \times N_\text{points} \times
        N_\text{dimensions}` where :math:`N_\text{points}` is the number of
        points specific to a cell (e.g. for a :class:`SimpleCell`, that is 2D
        quadrangle, the points are 4) and :math:`N_\text{dimensions}` is the
        dimensionality of the mesh, currently 2D
        (:math:`N_\text{dimensions}`=2)

    backend
        An instance of :class:`PlotBackend` used to plot mesh and its values
    """

    centroids: np.ndarray
    volumes: np.ndarray
    points: np.ndarray
    surfaces: np.ndarray
    normals: np.ndarray
    boundaries: Iterable[Boundary]

    # TODO: This will need to be removed when we go 3D
    MAX_DIMENSIONALITY = 2

    def __init__(
        self,
        left: BoundaryCurve,
        bottom: BoundaryCurve,
        right: BoundaryCurve,
        top: BoundaryCurve,
        cell_type: Type[Cell],
        Backend: Type[PlotBackend] = DefaultBackend,
    ):

        self.left = Boundary(left, (_BoundarySide.LEFT, slice(None)))
        self.btm = Boundary(bottom, (slice(None), _BoundarySide.BOTTOM))
        self.right = Boundary(right, (_BoundarySide.RIGHT, slice(None)))

        self.top = Boundary(top, (slice(None), _BoundarySide.TOP))

        self.cell_type = cell_type
        self.backend = Backend()

        self.boundaries = [
            boundary
            for boundary in (self.left, self.btm, self.right, self.top)
            if boundary.curve.bc is not None
        ]

        self.bcs_count = len(self.boundaries)

        self.dimensionality = self.bcs_count / 2

    def interpolate(
        self, num_cells_x: int, num_cells_y: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ This methods generates the mesh within the four given
        BoundaryCurve using Transfinite Interpolation

        Args:
            num_cells_x: Number of cells in :math:`x`-direction
            num_cells_y: Number of cells in :math:`y`-direction
        """

        self._num_xi = num_cells_x + 1
        self._num_eta = num_cells_y + 1
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y

        # This is the vectorized form of a double loop on xi and eta
        # to apply the TFI
        XIS, ETAS = np.ogrid[
            0 : 1 : self._num_xi * 1j,  # type: ignore
            0 : 1 : self._num_eta * 1j,  # type: ignore
        ]

        x = np.empty((self._num_xi, self._num_eta))
        y = np.empty((self._num_xi, self._num_eta))

        XL, YL = self.left.curve(ETAS)
        XR, YR = self.right.curve(ETAS)
        XB, YB = self.btm.curve(XIS)
        XT, YT = self.top.curve(XIS)
        XB0, YB0 = self.btm.curve(0)
        XB1, YB1 = self.btm.curve(1)
        XT0, YT0 = self.top.curve(0)
        XT1, YT1 = self.top.curve(1)

        x = (
            (1 - XIS) * XL
            + XIS * XR
            + (1 - ETAS) * XB
            + ETAS * XT
            - (1 - XIS) * (1 - ETAS) * XB0
            - (1 - XIS) * ETAS * XT0
            - (1 - ETAS) * XIS * XB1
            - XIS * ETAS * XT1
        )

        y = (
            (1 - XIS) * YL
            + XIS * YR
            + (1 - ETAS) * YB
            + ETAS * YT
            - (1 - XIS) * (1 - ETAS) * YB0
            - (1 - XIS) * ETAS * YT0
            - (1 - ETAS) * XIS * YB1
            - XIS * ETAS * YT1
        )

        self._x = x
        self._y = y

        # If we're doing a 1D simulation, we need to check that in the y
        # direction we have only one cell
        if self.dimensionality == 1:
            if self.num_cells_y > 1:
                raise InvalidMesh(
                    "The bottom and top BC are `None`. That means that you're "
                    "requesting a 1D simulation, hence in the y-direction you "
                    "need to set just 1 cell"
                )

            # Here I'm checking that each point on the bottom boundary has
            # same y-coordinate
            same_y = np.all((self._y[:, 0] - self._y[0, 0]) < 1e-12)

            # Same for the other row of points (we have two row of points and
            # one cell in the y-direction)
            same_y &= np.all((self._y[:, 1] - self._y[0, 1]) < 1e-12)

            if not (same_y):
                raise InvalidMesh(
                    "For a 1D simulation the top and bottom BoundaryCurve "
                    "needs to be a Line (i.e.  having same y-coordinate)"
                )

            # Let's scale dy in the y-direction to match dx in x-direction.
            # This is needed in order to have 2D flux to work also in 1D
            dx = self._x[1, 0] - self._x[0, 0]
            scale_y = self._y[0, 1] / dx
            self._y[:, 1] = self._y[:, 1] / scale_y

        return self._x, self._y

    def generate(self):
        """ Build the geometrical information and the connectivity associated
        to the mesh using the specific cell type
        :meth:`~.Cell.create_connectivity`
        """

        self.cell_type.create_connectivity(self)

    def export(self) -> MeshIO:
        """ Export the mesh to a :class:`meshio.Mesh` object """

        return self.cell_type.export_connectivity(self)

    def write(self, filepath: os.PathLike):
        """ Save the cell into a file using :mod:`meshio`

        Parameters
        ---------
        filepath
            The path to the file. The extension of the file is used by
            :mod:`meshio` to decide wich output file to use
        """

        meshio.write(filepath, self.export())

    def plot(self):
        """ This method shows the mesh in a GUI """

        self.backend.plot(self)
        self.backend.show_grid()
