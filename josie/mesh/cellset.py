# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Iterable, List

from josie.geometry import MeshIndex

if TYPE_CHECKING:
    from josie.boundary import Boundary
    from josie.dimension import Dimensionality
from josie.state import State


class NormalDirection(IntEnum):
    LEFT = 0
    RIGHT = 1
    BOTTOM = 2
    TOP = 3


class DimensionPair(Enum):
    X = [NormalDirection.LEFT, NormalDirection.RIGHT]
    Y = [NormalDirection.BOTTOM, NormalDirection.TOP]


@dataclass
class CellSet:
    r"""A dataclass representing a set of cells. It ships the values of the
    fields in the cells, together with the cell centroids and cell volumes, the
    normals to each face, the surface area of each face, and the field values


    Attributes
    ----------
    centroids
        An array containing the centroid of the cells. It has the dimensions of
        :math:`N_x \times N_y`

    volumes
        An array containing the volumes of the cells. It has the dimensions of
        :math:`N_x \times N_y`

    normals
        A :class:`np.ndarray` that has the dimensions :math:`Nx \times Ny
        \times N_\text{centroids} \times N_\text{dim}` containing the values of
        the normals to the faces of the cells

    surfaces
        An array containing the surfaces of the cells. It has the dimensions of
        :math:`N_x \times N_y \times N_\text{points}` where
        :math:`N_\text{points}` depends on the :class:`Cell` type provided

    values
        An array of dimensions :math:`N_x \times N_y \times N_\text{fields}`
        storing the value of the :class:`State` for each cell of the
        :class:`Mesh`

    dimensionality
        The :class:`Dimensionality` of the :class:`CellSet`

    min_length
        The minimal length of the :class:`CellSet`
    """

    volumes: np.ndarray
    surfaces: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray
    values: State
    dimensionality: Dimensionality
    min_length: float = np.nan

    def __getitem__(self, key: MeshIndex) -> CellSet:
        """Slice the :class:`CellSet` as a whole"""

        return CellSet(
            volumes=self.volumes[key],
            surfaces=self.surfaces[key],
            normals=self.normals[key],
            centroids=self.centroids[key],
            values=self.values[key].view(State),
            dimensionality=self.dimensionality,
        )

    def copy(self) -> CellSet:
        """Implement a fast copy to avoid using :mod:`copy.deepcopy`"""

        return CellSet(
            volumes=self.volumes.copy(),
            surfaces=self.surfaces.copy(),
            normals=self.normals.copy(),
            centroids=self.centroids.copy(),
            values=self.values.copy(),
            dimensionality=self.dimensionality,
        )

    def compute_min_length(self):
        """This method computes the min dx of the :class:`CellSet`.

        Useful for CFL computations
        """
        self.min_length = np.min(self.volumes[..., np.newaxis] / self.surfaces)


class NeighboursCellSet(CellSet):
    """A :class:`CellSet` that stores also the neighbours direction"""

    def __init__(
        self,
        volumes,
        surfaces,
        normals,
        centroids,
        values,
        direction,
        dimensionality,
    ):
        super().__init__(volumes, surfaces, normals, centroids, values, dimensionality)

        self.direction = direction


@dataclass
class MeshCellSetIndex:
    """A composite index used to slice the internal data structures of
    :class:`MeshCellSet`. It features a :class:`tuple` index to slice all
    the internal data structure and an additional :attr:`normal_direction` to
    choose the right normal direction

    Attributes
    ----------
    data_index
        The index to slice data structures containing also ghost cells data
    normal_direction
        The :class:`NormalDirection` to index the face-related data structures
    """

    data_index: MeshIndex
    normal_direction: NormalDirection


class NeighbourDirection(Enum):
    """An :class:`Enum` incapsulating the indexes for each neighbour
    direction to be used to slice :class:`MeshCellSet`"""

    LEFT = MeshCellSetIndex(
        data_index=(slice(-2), slice(1, -1)),
        normal_direction=NormalDirection.LEFT,
    )

    RIGHT = MeshCellSetIndex(
        data_index=(slice(2, None), slice(1, -1)),
        normal_direction=NormalDirection.RIGHT,
    )

    TOP = MeshCellSetIndex(
        data_index=(slice(1, -1), slice(2, None)),
        normal_direction=NormalDirection.TOP,
    )

    BOTTOM = MeshCellSetIndex(
        data_index=(slice(1, -1), slice(-2)),
        normal_direction=NormalDirection.BOTTOM,
    )


class MeshCellSet(CellSet):
    r"""A class representing the cells in a :class:`Mesh` object, i.e. a
    structured mesh. It stores internally the mesh cells data (centroids,
    values, etc...) including values for the ghost cells. The non-ghost values
    are then exposed as views on the internal array

    Attributes
    ----------
    centroids
        An array containing the centroid of the cells. It has the dimensions of
        :math:`N_x \times N_y`

    volumes
        An array containing the volumes of the cells. It has the dimensions of
        :math:`N_x \times N_y`

    normals
        A :class:`np.ndarray` that has the dimensions :math:`Nx \times Ny
        \times N_\text{centroids} \times N_\text{dim}` containing the values of
        the normals to the faces of the cells

    surfaces
        An array containing the surfaces of the cells. It has the dimensions of
        :math:`N_x \times N_y \times N_\text{points}` where
        :math`N_\text{points}` depends on the :class:`Cell` type provided

    values
        An array of dimensions :math:`N_x \times N_y \times N_\text{fields}`
        storing the value of the :class:`State` for each cell of the
        :class:`Mesh`

    min_length
        The minimal :math:`\dd{x}` of the mesh.

        Useful for CFL computations

    dimensionality
        The :class:`Mesh` dimensionality

    neighbours
        A list of :class:`NeighboursCellSet` representing the neighbours in a
        direction of all the cells
    """

    _values: State
    neighbours: List[NeighboursCellSet]

    def __init__(
        self,
        centroids: np.ndarray,
        volumes: np.ndarray,
        surfaces: np.ndarray,
        normals: np.ndarray,
        dimensionality: Dimensionality,
    ):
        self._centroids = centroids
        self._volumes = volumes
        self._surfaces = surfaces
        self._normals = normals
        self.dimensionality = dimensionality

    def __getitem__(self, key: MeshIndex) -> CellSet:
        """For :class:`MeshCellSet`, differently from :class:`CellSet`, we
        index on the internal data structures (i.e. the ones that contain also
        the ghost cells"""

        return CellSet(
            volumes=self._volumes[key],
            surfaces=self._volumes[key],
            normals=self._normals[key],
            centroids=self._centroids[key],
            values=self._values[key].view(State),
            dimensionality=self.dimensionality,
        )

    def copy(self) -> MeshCellSet:
        """Implement a fast copy to avoid using :mod:`copy.deepcopy`"""
        cells = MeshCellSet(
            centroids=self._centroids.copy(),
            volumes=self._volumes.copy(),
            surfaces=self._surfaces.copy(),
            normals=self._normals.copy(),
            dimensionality=self.dimensionality,
        )

        if self._values is not None:
            cells._values = self._values.copy()

        cells.create_neighbours()

        return cells

    def create_neighbours(self):
        directions = []
        if self.dimensionality > 0:
            directions = [
                NeighbourDirection.LEFT,
                NeighbourDirection.RIGHT,
            ]

        if self.dimensionality > 1:
            directions.extend((NeighbourDirection.BOTTOM, NeighbourDirection.TOP))

        # TODO Add 3D
        # if self.dimensionality > 2:

        self.neighbours = []

        for direction in directions:
            data_index = direction.value.data_index
            normal_direction = direction.value.normal_direction

            self.neighbours.append(
                NeighboursCellSet(
                    centroids=self._centroids[data_index],
                    values=self._values[data_index],
                    volumes=self._volumes[data_index],
                    normals=self.normals[..., normal_direction, :],
                    surfaces=self.surfaces[..., normal_direction],
                    dimensionality=self.dimensionality,
                    direction=normal_direction,
                )
            )

    @property  # type: ignore
    def volumes(self) -> np.ndarray:  # type: ignore
        return self._volumes[1:-1, 1:-1]

    @volumes.setter
    def volumes(self, value: np.ndarray):
        self._volumes[1:-1, 1:-1] = value

    @property  # type: ignore
    def surfaces(self) -> np.ndarray:  # type: ignore
        return self._surfaces[1:-1, 1:-1]

    @surfaces.setter
    def surfaces(self, value: np.ndarray):
        self._surfaces[1:-1, 1:-1] = value

    @property  # type: ignore
    def normals(self) -> np.ndarray:  # type: ignore
        return self._normals[1:-1, 1:-1]

    @normals.setter
    def normals(self, value: np.ndarray):
        self._normals[1:-1, 1:-1] = value

    @property  # type: ignore
    def centroids(self) -> np.ndarray:  # type: ignore
        return self._centroids[1:-1, 1:-1]

    @centroids.setter
    def centroids(self, value: np.ndarray):
        self._centroids[1:-1, 1:-1] = value

    @property  # type: ignore
    def values(self) -> State:  # type: ignore
        return self._values[1:-1, 1:-1].view(State)

    @values.setter
    def values(self, values: State):
        self._values[1:-1, 1:-1, :] = values

    def init_bcs(self, boundaries: Iterable[Boundary]):
        """Inits the data structures used by the :class:`BoundaryCondition`
        objects attached to each :class:`Boundary`

        Parameters
        ----------
        boundaries
            The :class:`Boundary` objects holding the
            :class:`BoundaryCondition` callables
        """
        for boundary in boundaries:
            boundary.init_bc(self)

    def update_ghosts(self, boundaries: Iterable[Boundary], t: float):
        """This method updates the ghost cells of the mesh with the current
        values depending on the specified boundary condition

        Parameters
        ----------
        boundaries
            The :class:`Boundary` objects holding the
            :class:`BoundaryCondition` callables
        t
            The time instant to evaluate time dependent
            :class:`BoundaryCondition`
        """

        for boundary in boundaries:
            boundary.apply_bc(self, t)
