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
from __future__ import annotations

import meshio
import numpy as np
import os

from meshio import Mesh as MeshIO
from typing import Iterable, Tuple, Type, TYPE_CHECKING

from josie.dimension import Dimensionality
from josie.exceptions import InvalidMesh
from josie.boundary import Boundary, BoundaryCurve, BoundarySide
from josie.plot import DefaultBackend
from josie.plot.backend import PlotBackend

if TYPE_CHECKING:
    from .cell import Cell
    from .cellset import MeshCellSet


class Mesh:
    r"""This class handles the mesh generation over a domain.

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

    bottom
        The bottom :class:`Boundary`

    right
        The right :class:`Boundary`

    top
        The right :class:`Boundary`

    boundaries
        An iterable over the boundaries available based on the
        :attr:`dimensionality`

    cell_type
        A :class:`Cell` class that implements a
        :func:`~Cell.create_connectivity`

    dimensionality
        The mesh dimensionality (i.e. 1 for 1D, 2 for 2D...)

    num_cells_x
        The number of cells in the :math:`x`-direction

    num_cells_y
        The number of cells in the :math:`y`-direction

    cells
        A :class:`MeshCellSet` storing cell data (centroids coordinates,
        normals to each face, face surface area, cell volume, field values)

    points
        An array containing the points that constitute a cell. It has the
        dimensions of :math:`N_x \times N_y \times N_\text{points} \times
        N_\text{dimensions}` where :math:`N_\text{points}` is the number of
        points specific to a cell (e.g. for a :class:`SimpleCell`, that is 2D
        quadrangle, the points are 4) and :math:`N_\text{dimensions}` is the
        dimensionality of the mesh, currently 2D
        (:math:`N_\text{dimensions} = 2`)

    backend
        An instance of :class:`PlotBackend` used to plot mesh and its values

    Note
    ----
    The centroids coordinates are stored internally in :attr:`_centroids`, an
    augmented array that also stores the centroids of the ghost cells
    """

    points: np.ndarray
    cells: MeshCellSet
    min_length: float
    boundaries: Iterable[Boundary]

    def __init__(
        self,
        left: BoundaryCurve,
        bottom: BoundaryCurve,
        right: BoundaryCurve,
        top: BoundaryCurve,
        cell_type: Type[Cell],
        Backend: Type[PlotBackend] = DefaultBackend,
    ):

        self.left = Boundary(
            side=BoundarySide.LEFT,
            curve=left,
            cells_idx=(BoundarySide.LEFT + 1, slice(1, -1)),
            ghost_cells_idx=(BoundarySide.LEFT, slice(1, -1)),
        )
        self.bottom = Boundary(
            side=BoundarySide.BOTTOM,
            curve=bottom,
            cells_idx=(slice(1, -1), BoundarySide.BOTTOM + 1),
            ghost_cells_idx=(slice(1, -1), BoundarySide.BOTTOM),
        )
        self.right = Boundary(
            side=BoundarySide.RIGHT,
            curve=right,
            cells_idx=(BoundarySide.RIGHT - 1, slice(1, -1)),
            ghost_cells_idx=(BoundarySide.RIGHT, slice(1, -1)),
        )

        self.top = Boundary(
            side=BoundarySide.TOP,
            curve=top,
            cells_idx=(slice(1, -1), BoundarySide.TOP - 1),
            ghost_cells_idx=(slice(1, -1), BoundarySide.TOP),
        )

        self.cell_type = cell_type
        self.backend = Backend()

        self.boundaries = [
            boundary
            for boundary in (self.left, self.bottom, self.right, self.top)
            if boundary.curve.bc is not None
        ]

        self.bcs_count = len(self.boundaries)

        self.dimensionality = Dimensionality(self.bcs_count / 2)

    def copy(self):
        """ This methods copies the :class:`Mesh` object into another """

        mesh = Mesh(
            self.left,
            self.bottom,
            self.right,
            self.top,
            self.cell_type,
            type(self.backend),
        )

        # Copy stateful attributes
        mesh.cells = self.cells.copy()
        mesh.points = self.points.copy()

        return mesh

    def interpolate(
        self, num_cells_x: int, num_cells_y: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """This methods generates the mesh within the four given
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
        XB, YB = self.bottom.curve(XIS)
        XT, YT = self.top.curve(XIS)
        XB0, YB0 = self.bottom.curve(0)
        XB1, YB1 = self.bottom.curve(1)
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
        if self.dimensionality is Dimensionality.ONED:
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

    def create_neighbours(self):
        """This is a proxy method to :meth:`~MeshCellSet.create_neighbours`
        that creates the internal connectivity for cell neighbours"""

        self.cells.create_neighbours()

    def update_ghosts(self, t: float):
        """This method updates the ghost cells of the mesh with the current
        values depending on the specified boundary condition

        Parameters
        ----------
        t
            The time instant to evaluate time dependent
            :class:`BoundaryCondition`
        """

        self.cells.update_ghosts(self.boundaries, t)

    def generate(self):
        """Build the geometrical information and the connectivity associated
        to the mesh using the specific cell type
        :meth:`~.Cell.create_connectivity`
        """

        self.cell_type.create_connectivity(self)

    def export(self) -> MeshIO:
        """ Export the mesh to a :class:`meshio.Mesh` object """

        return self.cell_type.export_connectivity(self)

    def write(self, filepath: os.PathLike):
        """Save the cell into a file using :mod:`meshio`

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
