# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc

import numpy as np

from meshio import Mesh as MeshIO
from typing import TYPE_CHECKING

from josie.geometry import PointType
from josie.dimension import MAX_DIMENSIONALITY
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
    def num_points(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_dofs(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _meshio_cell_type(self) -> str:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _generate_ghosts(cls, mesh: Mesh):
        """Generate ghost cells centroids"""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def centroid(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> PointType:
        """Compute the centroid of the cell"""

    @classmethod
    @abc.abstractmethod
    def volume(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> np.ndarray:
        """Compute the volume of a cell from its points."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def face_surface(cls, p0: PointType, p1: PointType) -> np.ndarray:
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

        By default it just calls :meth:`Mesh.cells.compute_min_length`. When
        subclassing, create you connectivity first, than call
        :meth`super().create_connectivity`

        It takes into account the nature of the cell composing the
        mesh, e.g. quadrangles.

        **Data Structure**

        The data structures holding the cell centroids and field values on the
        mesh are :class:`numpy.ndarray` of dimensions :math:`N_x + 2 \times N_y
        + 2\times \ldots`. That is, we have one layer of ghosts per direction.
        (The corner cells are unused)

        Parameters
        ----------
        mesh
            An instance of the :class:`Mesh` of which we need to create the
            connectivity

        """

        cls._generate_ghosts(mesh)
        mesh.cells.compute_min_length()

    @classmethod
    @abc.abstractmethod
    def export_connectivity(cls, mesh: "Mesh") -> MeshIO:
        """This method exports the connectivity of the mesh in the format
        accepted by the :class:`~meshio.Mesh`.
        """
        raise NotImplementedError


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

        return np.asarray(nw + sw + se + ne) / 4

    @classmethod
    def volume(
        cls, nw: PointType, sw: PointType, se: PointType, ne: PointType
    ) -> np.ndarray:
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

        volume = (
            np.linalg.norm(np.cross(sw - nw, se - sw)[..., np.newaxis], axis=-1) / 2
        )

        volume = (
            volume
            + np.linalg.norm(np.cross(se - ne, ne - nw)[..., np.newaxis], axis=-1) / 2
        )

        return volume

    @classmethod
    def face_surface(cls, p0: PointType, p1: PointType) -> np.ndarray:
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

        return np.linalg.norm(p0 - p1, axis=-1)

    @classmethod
    def face_normal(cls, p0: PointType, p1: PointType) -> np.ndarray:
        r"""This class method computes the normal to a face from its points.

        The normal is computed as the ortogonal vector to the vector made
        by the two given points obtained doing a CW rotation of 90 degrees

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

        normal = np.transpose(np.array([r[..., 1], -r[..., 0]]), (1, 2, 0))

        return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    @classmethod
    def create_connectivity(cls, mesh: Mesh):
        nx = mesh.num_cells_x
        ny = mesh.num_cells_y
        x = mesh._x
        y = mesh._y
        dimensionality = mesh.dimensionality

        # TODO Get mesh.dimensionality as input and adapt the size of the
        # containers accordingly

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
        normals = np.full((nx + 2, ny + 2, cls.num_points, MAX_DIMENSIONALITY), np.nan)
        surfaces = np.full((nx + 2, ny + 2, cls.num_points), np.nan)

        cells = MeshCellSet(
            centroids=centroids,
            volumes=volumes,
            surfaces=surfaces,
            normals=normals,
            dimensionality=dimensionality,
            min_length=np.nan,
        )

        # Loop to build connectivity
        nw = np.transpose(np.asarray((x[:-1, 1:], y[:-1, 1:])), (1, 2, 0))
        sw = np.transpose(np.asarray((x[:-1, :-1], y[:-1, :-1])), (1, 2, 0))
        se = np.transpose(np.asarray((x[1:, :-1], y[1:, :-1])), (1, 2, 0))
        ne = np.transpose(np.asarray((x[1:, 1:], y[1:, 1:])), (1, 2, 0))

        mesh.points = np.stack((nw, sw, se, ne), axis=-2)
        cells.centroids = np.asarray(cls.centroid(nw, sw, se, ne))[..., np.newaxis, :]
        cells.volumes = cls.volume(nw, sw, se, ne)
        cells.surfaces[..., NormalDirection.LEFT] = cls.face_surface(nw, sw)
        cells.surfaces[..., NormalDirection.BOTTOM] = cls.face_surface(sw, se)
        cells.surfaces[..., NormalDirection.RIGHT] = cls.face_surface(se, ne)
        cells.surfaces[..., NormalDirection.TOP] = cls.face_surface(ne, nw)
        cells.normals[..., NormalDirection.LEFT, :] = cls.face_normal(nw, sw)
        cells.normals[..., NormalDirection.BOTTOM, :] = cls.face_normal(sw, se)
        cells.normals[..., NormalDirection.RIGHT, :] = cls.face_normal(se, ne)
        cells.normals[..., NormalDirection.TOP, :] = cls.face_normal(ne, nw)

        # Assign back to mesh object
        mesh.cells = cells

        super().create_connectivity(mesh)

    @classmethod
    def _generate_ghosts(cls, mesh: Mesh):
        r"""
        Generate ghost cells centroids ortogonal to
        :math:\hat{\vb{x}},\hat{\vb{y}},\hat{\vb{z}}
        at unitary distance

        """

        mesh.cells.compute_min_length()

        for boundary in mesh.boundaries:
            side = boundary.side
            boundary_idx = boundary.cells_idx
            ghost_idx = boundary.ghost_cells_idx

            boundary_centroids = mesh.cells._centroids[boundary_idx[0], boundary_idx[1]]

            # Compute the ghost cells centroids
            mesh.cells._centroids[ghost_idx[0], ghost_idx[1]] = (
                boundary_centroids + cls._side_normal[side.name] * mesh.cells.min_length
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
        num_chunks = int(rows / cls.num_points)
        io_cells = np.split(np.arange(rows), num_chunks)

        return MeshIO(io_pts, {cls._meshio_cell_type: np.array(io_cells)})


class MUSCLCell(SimpleCell):
    """
    The first DOF is in the center of cell
    The next four are the extrapolated values for MUSCL at the center of each faces
    """

    num_dofs = 5
