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

from enum import IntEnum

import numpy as np

from meshio import Mesh as MeshIO
from typing import TYPE_CHECKING, Tuple, Union


if TYPE_CHECKING:
    from josie.mesh import Mesh


PointType = Union[Tuple[float, float], np.ndarray]


class NormalDirection(IntEnum):
    LEFT = 0
    BOTTOM = 1
    RIGHT = 2
    TOP = 3


class Cell(metaclass=abc.ABCMeta):
    """ This is a class interface representing a generic cell of a
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
        """ Compute the volume of a cell from its points.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def face_surface(cls, p0: PointType, p1: PointType) -> float:
        """ Compute the surface of a face from its points.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def face_normal(cls, p0: PointType, p1: PointType) -> np.ndarray:
        """ Compute the normal vector to a face.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def create_connectivity(cls, mesh: Mesh):
        """ This method creates the connectivity from the given points of
        a mesh. It modifies attributes of the :class:`Mesh` instance.

        It takes into account the nature of the cell composing the
        mesh, e.g. quadrangles.

        Parameters
        ----------
        x
            x-coordinates of the points of the mesh
        y
            y-coordinates of the points of the mesh

        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def export_connectivity(cls, mesh: "Mesh") -> MeshIO:
        """ This method exports the connectivity of the mesh in the format
        accepted by the :class:`~meshio.Mesh`.
        """


class SimpleCell(Cell):
    """ This class describes the classical type of 2D quadrangular cell that
    stores the :class:`State` value in its centroid. The cell needs 4 points
    to be defined and has 1 degree of freedom.

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

    @classmethod
    def centroid(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> PointType:
        """ This class method computes the centroid of a cell from its points.

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
        """ This class method computes the volume of a cell from its points.

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
        """ This class method computes the surface of a face from its points.

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
    def create_connectivity(cls, mesh: "Mesh"):
        num_cells_x = mesh.num_cells_x
        num_cells_y = mesh.num_cells_y
        x = mesh._x
        y = mesh._y

        # This are the centroids of the cells. For cell i, j the third
        # dimension contain the coordinates of each centroid of a cell.
        # So in the SimpleCell case, each cell stored one DOF, so
        # it has a 3rd dimension of 1 (== num_dofs) and a 4th dimension
        # equal to 2 (since each coordinate point has x, y coordinates)
        mesh.points = np.empty((num_cells_x, num_cells_y, cls.num_points, 2))
        mesh.centroids = np.empty((num_cells_x, num_cells_y, 2))
        mesh.volumes = np.empty((num_cells_x, num_cells_y))
        mesh.normals = np.empty_like(mesh.points)
        mesh.surfaces = np.empty((num_cells_x, num_cells_y, cls.num_points))

        # TODO: This can be probably vectorized
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                p0 = np.asarray((x[i, j + 1], y[i, j + 1]))
                p1 = np.asarray((x[i, j], y[i, j]))
                p2 = np.asarray((x[i + 1, j], y[i + 1, j]))
                p3 = np.asarray((x[i + 1, j + 1], y[i + 1, j + 1]))

                mesh.points[i, j, :, :] = np.vstack(
                    (p0, p1, p2, p3)
                )  # type: ignore # noqa: E501
                mesh.centroids[i, j, :] = cls.centroid(
                    p0, p1, p2, p3
                )  # type: ignore # noqa: E501
                mesh.volumes[i, j] = cls.volume(
                    p0, p1, p2, p3
                )  # type: ignore # noqa: E501

                mesh.surfaces[i, j, NormalDirection.LEFT] = cls.face_surface(
                    p0, p1
                )  # type: ignore # noqa: E501
                mesh.surfaces[i, j, NormalDirection.BOTTOM] = cls.face_surface(
                    p1, p2
                )  # type: ignore # noqa: E501
                mesh.surfaces[i, j, NormalDirection.RIGHT] = cls.face_surface(
                    p2, p3
                )  # type: ignore # noqa: E501
                mesh.surfaces[i, j, NormalDirection.TOP] = cls.face_surface(
                    p3, p0
                )  # type: ignore # noqa: E501

                mesh.normals[i, j, NormalDirection.LEFT, :] = cls.face_normal(
                    p0, p1
                )  # type: ignore # noqa: E501
                mesh.normals[
                    i, j, NormalDirection.BOTTOM, :
                ] = cls.face_normal(
                    p1, p2
                )  # type: ignore # noqa: E501

                mesh.normals[i, j, NormalDirection.RIGHT, :] = cls.face_normal(
                    p2, p3
                )  # type: ignore # noqa: E501
                mesh.normals[i, j, NormalDirection.TOP, :] = cls.face_normal(
                    p3, p0
                )  # type: ignore # noqa: E501

    @classmethod
    def export_connectivity(cls, mesh: "Mesh") -> MeshIO:
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
