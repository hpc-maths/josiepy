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
from __future__ import annotations

import abc

import numpy as np

from meshio import Mesh as MeshIO
from typing import TYPE_CHECKING

from josie.geom import PointType
from josie._dim import MAX_DIMENSIONALITY
from josie.math import R3

from .cellset import MeshCellSet, NormalDirection


if TYPE_CHECKING:
    from josie.mesh import Mesh


class Cell(metaclass=abc.ABCMeta):
    """This is a class interface representing a generic cell of a
    :class:`Mesh`.

    A cell is defined by the number of points (:attr:`num_points`) that are
    needed to properly define it and the number of degrees of freedom
    (:attr:`num_dofs`) actually availbe within it. It needs to provide also the
    methods to compute geometrical informations (e.g. its volume, normals, face
    area, etc...).

    Attributes
    ---------
    num_points: int
        The number of points needed to describe the cell
    num_dofs: int
        The number of degrees of freedom stored in the cell
    _meshio_cell_type: str
        Which type of cell in :mod:`meshio` :class:`Cell` is mapped to (e.g.
        a quadrangular cell is of type `quad` in :mod:`meshio`
    """

    @property
    @abc.abstractmethod
    def num_points(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_dofs(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _meshio_cell_type(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def centroid(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> PointType:
        """ Compute the centroid of the cell """

    @classmethod
    @abc.abstractmethod
    def volume(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> float:
        """Compute the volume of a cell from its points."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def face_surface(cls, p0: PointType, p1: PointType) -> float:
        """Compute the surface of a face from its points."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def face_normal(cls, p0: PointType, p1: PointType) -> np.ndarray:
        """Compute the normal vector to a face."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def create_connectivity(cls, mesh: Mesh):
        r"""This method creates the connectivity from the given points of
        a mesh. It modifies attributes of the :class:`Mesh` instance.

        It takes into account the nature of the cell composing the
        mesh, e.g. quadrangles.

        **Data Structure**

        The data structures holding the cell centroids and field values on the
        mesh are :class:`numpy.ndarray` of dimensions :math:`N_x + 2 \times N_y
        + 2\times \ldots`. That is, we have one layer of ghosts per direction.
        (The corner cells are unused)

        Parameters
        ----------
        x
            :math:`x`-coordinates of the points of the mesh
        y
            :math:`y`-coordinates of the points of the mesh

        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def export_connectivity(cls, mesh: "Mesh") -> MeshIO:
        """This method exports the connectivity of the mesh in the format
        accepted by the :class:`~meshio.Mesh`.
        """


class SimpleCell(Cell):
    """This class describes the classical type of 2D quadrangular cell that
    stores the :class:`State` value in its centroid. The cell needs 4 points
    to be defined and has 1 degree of freedom.

    .. code-block::

        nw     ne
        *-------*
        |       |
        |   *   |
        |   c   |
        *-------*
        sw     se

    """

    num_dofs = 1
    num_points = 4
    _meshio_cell_type = "quad"

    _side_normal = {"LEFT": -R3.X, "RIGHT": R3.X, "TOP": R3.Y, "BOTTOM": -R3.Y}

    @classmethod
    def centroid(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> PointType:
        """This class method computes the centroid of a cell from its points.

        The centroid is computed as the mean value of the for points


        Parameters
        ----------
        nw
            The North-West point of the cell
        sw
            The South-West point of the cell
        se
            The South-East point of the cell
        ne
            The North-East point of the cell


        Returns
        -------
        centroid
            The centroid coordinates

        """
        nw = np.asarray(nw)
        sw = np.asarray(sw)
        se = np.asarray(se)
        ne = np.asarray(ne)

        return (nw + sw + se + ne) / 4

    @classmethod
    def volume(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> float:
        """This class method computes the volume of a cell from its points.

        The surface is computed calculating the surface of the two triangles
        made by the two triplets of its points and summing them up.

        Parameters
        ----------
        nw
            The North-West point of the cell
        sw
            The South-West point of the cell
        se
            The South-East point of the cell
        ne
            The North-East point of the cell

        Returns
        -------
        volume: The volume of the cell

        """

        nw = np.asarray(nw)
        sw = np.asarray(sw)
        se = np.asarray(se)
        ne = np.asarray(ne)

        volume = np.linalg.norm(np.cross(sw - nw, se - sw)) / 2

        volume = volume + np.linalg.norm(np.cross(se - ne, ne - nw)) / 2

        return volume

    @classmethod
    def face_surface(cls, p0: PointType, p1: PointType) -> float:
        """This class method computes the surface of a face from its points.

        The surface is simply the norm of the vector that is made by the two
        given points of the face (being in 2D).

        Parameters
        ----------
        p0
            The first point of the face
        p1
            The second point of the face

        Returns:
            surface: The "surface" (i.e. the length) of the face
        """
        p0 = np.asarray(p0)
        p1 = np.asarray(p1)

        return np.linalg.norm(p0 - p1)

    @classmethod
    def face_normal(cls, p0: PointType, p1: PointType) -> np.ndarray:
        r""" This class method computes the normal to a face from its points.

        The normal is computed as the ortogonal vector to the vector made
        by the two given points obtained doing a CW rotation

        .. todo::

            Add image with TiKz

        Parameters
        ----------
        p0
            The first point of the face
        p1
            The second point of the face

        Returns
        -------
        normal
            The normal vector to the face
        """
        p0 = np.asarray(p0)
        p1 = np.asarray(p1)

        r = p1 - p0

        normal = np.array([r[1], -r[0]])

        return normal / np.linalg.norm(normal)

    @classmethod
    def create_connectivity(cls, mesh: Mesh):
        nx = mesh.num_cells_x
        ny = mesh.num_cells_y
        x = mesh._x
        y = mesh._y

        # Init internal data structures
        points = np.empty((nx, ny, cls.num_points, MAX_DIMENSIONALITY))

        # Create basic :class:`MeshCellSet` data structure that contains data
        # also for the ghost cells. Init with NaN since some of the entries are
        # not used for ghost cells (i.e. corner values for centroids and
        # values, entire array for volumes and surfaces)

        centroids = np.full(
            (
                nx + 2,
                ny + 2,
                cls.num_dofs,
                MAX_DIMENSIONALITY,
            ),
            np.nan,
        )

        volumes = np.full((nx + 2, ny + 2), np.nan)
        normals = np.full(
            (nx + 2, ny + 2, cls.num_points, MAX_DIMENSIONALITY), np.nan
        )
        surfaces = np.full((nx + 2, ny + 2, cls.num_points), np.nan)

        cells = MeshCellSet(
            centroids=centroids,
            volumes=volumes,
            surfaces=surfaces,
            normals=normals,
        )

        # Loop to build connectivity
        # TODO: This can be probably vectorized
        for i in range(nx):
            for j in range(ny):
                p0 = np.asarray((x[i, j + 1], y[i, j + 1]))
                p1 = np.asarray((x[i, j], y[i, j]))
                p2 = np.asarray((x[i + 1, j], y[i + 1, j]))
                p3 = np.asarray((x[i + 1, j + 1], y[i + 1, j + 1]))

                points[i, j, :, :] = np.vstack(
                    (p0, p1, p2, p3)
                )  # type: ignore # noqa: E501
                cells.centroids[i, j, :] = cls.centroid(
                    p0, p1, p2, p3
                )  # type: ignore # noqa: E501
                cells.volumes[i, j] = cls.volume(
                    p0, p1, p2, p3
                )  # type: ignore # noqa: E501

                cells.surfaces[i, j, NormalDirection.LEFT] = cls.face_surface(
                    p0, p1
                )  # type: ignore # noqa: E501
                cells.surfaces[
                    i, j, NormalDirection.BOTTOM
                ] = cls.face_surface(
                    p1, p2
                )  # type: ignore # noqa: E501
                cells.surfaces[i, j, NormalDirection.RIGHT] = cls.face_surface(
                    p2, p3
                )  # type: ignore # noqa: E501
                cells.surfaces[i, j, NormalDirection.TOP] = cls.face_surface(
                    p3, p0
                )  # type: ignore # noqa: E501

                cells.normals[i, j, NormalDirection.LEFT, :] = cls.face_normal(
                    p0, p1
                )  # type: ignore # noqa: E501
                cells.normals[
                    i, j, NormalDirection.BOTTOM, :
                ] = cls.face_normal(
                    p1, p2
                )  # type: ignore # noqa: E501

                cells.normals[
                    i, j, NormalDirection.RIGHT, :
                ] = cls.face_normal(
                    p2, p3
                )  # type: ignore # noqa: E501
                cells.normals[i, j, NormalDirection.TOP, :] = cls.face_normal(
                    p3, p0
                )  # type: ignore # noqa: E501

        # Assign back to mesh object
        mesh.points = points
        mesh.cells = cells

        cls._generate_ghosts(mesh)

    @classmethod
    def _generate_ghosts(cls, mesh: Mesh):
        r"""
        Generate ghost cells centroids ortogonal to
        :math:\hat{\vb{x}},\hat{\vb{y}},\hat{\vb{z}}

        """
        for boundary in mesh.boundaries:
            side = boundary.side
            boundary_idx = boundary.cells_idx
            ghost_idx = boundary.ghost_cells_idx

            boundary_centroids = mesh.cells._centroids[
                boundary_idx[0], boundary_idx[1]
            ]

            # Compute the ghost cells centroids
            mesh.cells._centroids[ghost_idx[0], ghost_idx[1]] = (
                boundary_centroids + cls._side_normal[side.name]
            )

    @classmethod
    def export_connectivity(cls, mesh: Mesh) -> MeshIO:
        # Recast points in the format meshio wants them. I.e. an array of
        # [nx*ny*num_points, 3] elements. Each row is a point in 3D.
        nx = mesh.num_cells_x
        ny = mesh.num_cells_y

        # 2D points
        io_pts = mesh.points.reshape(nx * ny * cls.num_points, 2)

        # Pad with zeros to make them 3D
        io_pts = np.pad(io_pts, ((0, 0), (0, 1)))

        # Now a cell is made by chunks of `num_points` rows of the io_pts
        # array
        rows, _ = io_pts.shape
        num_chunks = rows / cls.num_points
        io_cells = np.split(np.arange(rows), num_chunks)

        return MeshIO(io_pts, {cls._meshio_cell_type: np.array(io_cells)})
