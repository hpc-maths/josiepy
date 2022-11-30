# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import abc

import copy
from josie.mesh.cellset import MeshCellSet, NeighboursCellSet

from josie.scheme.convective import ConvectiveScheme

from josie.mesh.cellset import DimensionPair

from josie.fluid.state import ConsState


class MUSCL_Hancock(ConvectiveScheme):
    _values: ConsState

    slopes: np.ndarray

    # Parameter for limiters
    # between -1 and 1
    omega = 0

    @abc.abstractmethod
    def compute_slopes(self, cells: MeshCellSet):
        r"""Compute the slopes of the local linear approximation of
        neighbouring state values. Limiters can be used here to limit the
        oscillations of linear approximation in the neighbourhood of
        discontinuities.

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` which contains mesh info such as normals,
            volumes or surfaces of the cells/interfaces.
        """
        pass

    def update_values_face(self, cells: MeshCellSet, dt: float):
        r"""Updates the extrapolated values at each interface for half
        a timestep using flux evaluated within the cell.

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` which contains mesh info such as normals,
            volumes or surfaces of the cells/interfaces.

        dt
            A `float` to store the timestep.
        """

        for i, dim in enumerate(DimensionPair):
            if i >= cells.dimensionality:
                break
            dir_L = dim.value[0].value
            dir_R = dim.value[1].value
            neigh_L = cells.neighbours[dir_L]
            neigh_R = cells.neighbours[dir_R]

            n_L = neigh_L.normals
            n_R = neigh_R.normals

            Q_L = self.values_face.values[..., dir_L]
            Q_R = self.values_face.values[..., dir_R]

            F_L = np.einsum("...mkl,...l->...mk", self.problem.F(Q_L), n_L)
            F_R = np.einsum("...mkl,...l->...mk", self.problem.F(Q_R), n_R)

            state_cls = cells._values.__class__
            Q_L.view(state_cls).set_conservative(  # type: ignore
                Q_L.view(state_cls).get_conservative()  # type: ignore
                - 0.5
                * dt
                / cells.volumes[..., np.newaxis, np.newaxis]
                * cells.surfaces[..., np.newaxis, [dir_L]]
                * (F_L + F_R)
            )

            Q_R.view(state_cls).set_conservative(  # type: ignore
                Q_R.view(state_cls).get_conservative()  # type: ignore
                - 0.5
                * dt
                / cells.volumes[..., np.newaxis, np.newaxis]
                * cells.surfaces[..., np.newaxis, [dir_R]]
                * (F_L + F_R)
            )

    def pre_extrapolation(self, cells: MeshCellSet):
        r"""Optional step before applying the slopes to compute the state
        values at each face. For instance, this is where the additional
        Berthon limiter is implemented.

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` which contains mesh info such as normals,
            volumes or surfaces of the cells/interfaces.
        """
        pass

    def linear_extrapolation(self, cells: MeshCellSet):
        # Compute linear extrapolated values at each face
        for direction in range(2**cells.dimensionality):
            self.values_face.values[..., direction] = (
                cells.values + 0.5 * self.slopes[..., direction]
            )

    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet):
        # Solve the Riemann problem to compute the intercell flux
        # using the extrapolated and half-timestep updated states of each
        # interface as initial conditions.

        direction = neighs.direction
        oppDirection = direction + 1 if direction % 2 == 0 else direction - 1
        Q_L = self.values_face.values[..., direction]
        Q_R = self.values_face.neighbours[direction].values[..., oppDirection]

        return self.intercellFlux(
            Q_L,
            Q_R,
            neighs.normals,
            neighs.surfaces,
        )

    def post_init(self, cells: MeshCellSet):
        r"""Initialize the datastructure holding the values at interface
        for each cell and face
        """

        super().post_init(cells)

        self.values_face = cells.copy()

        self.slopes = np.empty(cells.values.shape + (2**cells.dimensionality,)).view(
            cells._values.__class__
        )

        self.values_face._values = np.empty(
            cells._values.shape + (2**cells.dimensionality,)
        ).view(cells._values.__class__)

        self.values_face.create_neighbours()

    def pre_accumulate(self, cells: MeshCellSet, dt: float, t: float):
        super().pre_accumulate(cells, dt, t)

        self.slopes.fill(0)

        # Initialize state values at each face with the state value
        # of the cell
        for dir in range(2**cells.dimensionality):
            self.values_face._values[..., dir] = cells._values.copy()

        # Compute the slope for each direction according to the
        # chosen limiter
        # TODO : adapt the limiters for a case with a non-constant
        # space step
        self.compute_slopes(cells)

        # If needed (e.g. Berthon), apply complementary computations
        self.pre_extrapolation(cells)

        # Extrapolation of the cell value (the conserved ones) into
        # the value at each interface thanks to the computed slopes
        self.linear_extrapolation(cells)

        # Update the auxiliary components at each face
        for dir in range(2**cells.dimensionality):
            self.post_extrapolation(self.values_face._values[..., dir])

        # Perform the half-timestep at each interface using cell
        # flux (for the conserved components)
        # self.update_values_face(cells, dt)

        # Update the auxiliary components at each face
        # for dir in range(2**cells.dimensionality):
        #     self.post_extrapolation(self.values_face._values[..., dir])
