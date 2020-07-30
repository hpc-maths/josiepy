from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple

from josie.geom import MeshIndex
from josie.solver.state import State


class NormalDirection(IntEnum):
    LEFT = 0
    BOTTOM = 1
    RIGHT = 2
    TOP = 3


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
        N_\text{points} depends on the :class:`Cell` type provided

    values
        An array of dimensions :math:`N_x \times N_y \times N_\text{fields}`
        storing the value of the :class:`State` for each cell of the
        :class:`Mesh`
    """

    volumes: np.ndarray
    surfaces: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray
    values: State

    def __getitem__(self, key: Tuple[MeshIndex]) -> CellSet:
        """ Slice the :class:`CellSet` as a whole """

        return CellSet(
            volumes=self.volumes[key],
            surfaces=self.volumes[key],
            normals=self.normals[key],
            centroids=self.centroids[key],
            values=self.values[key],
        )


@dataclass
class MeshCellSetIndex:
    """A composite index used to slice the internal data structures of
    :class:`MeshCellSet`. It features a :class:`tuple` index to slice
    :attr:`MeshCellSet._centroids` and :attr:`MeshCellSet._values`, another
    tuple to slice :attr:`MeshCellSet.volumes` since it does not store volume
    values for ghost cells, and a :class:`NormalDirection` to index
    :attr:`MeshCellSet.normals` and :attr:`MeshCellSet.surfaces`
    """

    ghost_data_index: Tuple[MeshIndex, ...]
    true_data_index: Tuple[MeshIndex, ...]
    normal_direction: NormalDirection


class NeighbourDirection(Enum):
    """An :class:`Enum` incapsulating the indexes for each neighbour
    direction to be used to slice :class:`MeshCellSet`"""

    LEFT = MeshCellSetIndex(
        ghost_data_index=(slice(-2), slice(1, -1)),
        true_data_index=(0, slice(None)),
        normal_direction=NormalDirection.LEFT,
    )

    RIGHT = MeshCellSetIndex(
        ghost_data_index=(slice(2, None), slice(1, -1)),
        true_data_index=(-1, slice(None)),
        normal_direction=NormalDirection.RIGHT,
    )

    TOP = MeshCellSetIndex(
        ghost_data_index=(slice(1, -1), slice(2, None)),
        true_data_index=(slice(None), -1),
        normal_direction=NormalDirection.TOP,
    )

    BOTTOM = MeshCellSetIndex(
        ghost_data_index=(slice(1, -1), slice(-2)),
        true_data_index=(slice(None), 0),
        normal_direction=NormalDirection.BOTTOM,
    )


class MeshCellSet(CellSet):
    r""" A specific :class:`CellSet` representing the cells in a :class:`Mesh`
    object, i.e. a structured mesh. It stores internally the mesh cells data
    (centroids, values, etc...) including values for the ghost cells. The
    non-ghost values are then exposed as views on the internal array
    """

    _values: State

    def __init__(self, _centroids: np.ndarray, volumes, surfaces, normals):
        self._centroids = _centroids
        self.volumes = volumes
        self.surfaces = surfaces
        self.normals = normals

    def get_neighbours(self, key: NeighbourDirection) -> CellSet:
        """Similar to :meth:`__getitem__`, but to slice the
        :class:`MeshCellSet` as whole to get neighbours on all the mesh sides
        """
        ghost_data_index = key.value.ghost_data_index
        true_data_index = key.value.true_data_index
        normal_direction = key.value.normal_direction

        return CellSet(
            centroids=self._centroids[ghost_data_index],
            values=self._values[ghost_data_index],
            volumes=self.volumes[true_data_index],
            normals=self.normals[..., normal_direction, :],
            surfaces=self.surfaces[..., normal_direction],
        )

    @property  # type: ignore
    def centroids(self) -> np.ndarray:
        return self._centroids[1:-1, 1:-1]

    @centroids.setter
    def centroids(self, value: np.ndarray):
        self._centroids[1:-1, 1:-1] = value

    @property  # type: ignore
    def values(self) -> State:
        return self._values[1:-1, 1:-1]

    @values.setter
    def values(self, values: State):
        self._values[1:-1, 1:-1, :] = values
