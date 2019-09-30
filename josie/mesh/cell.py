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
import abc

import numpy as np

from enum import Enum
from typing import TYPE_CHECKING, Tuple, Union


if TYPE_CHECKING:
    from josie.solver.state import State  # noqa: F401
    from josie.mesh import Mesh  # noqa: F401  # noqa: F401


PointType = Union[Tuple[float, float], np.ndarray]


class Dimensionality(Enum):
    TWO = 2
    THREE = 3


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
    dims: Dimensionality
        Dimensionality of the cell.
    """

    @abc.abstractproperty
    @property
    def num_points(self):
        raise NotImplementedError

    @abc.abstractproperty
    @property
    def num_dofs(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractclassmethod
    def centroid(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> PointType:
        """ Compute the centroid of the cell """

    @classmethod
    @abc.abstractclassmethod
    def volume(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> float:
        """ Compute the volume of a cell from its points.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractclassmethod
    def face_surface(cls, p0: PointType, p1: PointType) -> float:
        """ Compute the surface of a face from its points.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractclassmethod
    def face_normal(cls, p0: PointType, p1: PointType) -> np.ndarray:
        """ Compute the normal vector to a face.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractclassmethod
    def create_connectivity(cls, mesh: "Mesh"):
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
    dims = Dimensionality.TWO

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
        by the two given points obtained doing a CCW rotation

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

                mesh.surfaces[i, j, 0] = cls.face_surface(
                    p0, p1
                )  # type: ignore # noqa: E501
                mesh.surfaces[i, j, 1] = cls.face_surface(
                    p1, p2
                )  # type: ignore # noqa: E501
                mesh.surfaces[i, j, 2] = cls.face_surface(
                    p2, p3
                )  # type: ignore # noqa: E501
                mesh.surfaces[i, j, 3] = cls.face_surface(
                    p0, p3
                )  # type: ignore # noqa: E501

                mesh.normals[i, j, 0, :] = cls.face_normal(
                    p0, p1
                )  # type: ignore # noqa: E501
                mesh.normals[i, j, 1, :] = cls.face_normal(
                    p1, p2
                )  # type: ignore # noqa: E501
                mesh.normals[i, j, 2, :] = cls.face_normal(
                    p2, p3
                )  # type: ignore # noqa: E501
                mesh.normals[i, j, 3, :] = cls.face_normal(
                    p0, p3
                )  # type: ignore # noqa: E501
