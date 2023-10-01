# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cellset import MeshCellSet, NeighboursCellSet, MUSCLMeshCellSet

from josie.scheme.convective import ConvectiveScheme

from josie.mesh.cellset import DimensionPair


class MUSCL(ConvectiveScheme):
    slopes: np.ndarray

    # Parameter for limiters
    # between -1 and 1
    omega = 0

    def compute_slopes(self, cells: MeshCellSet):
        # Compute intercell slopes for each face
        # We assume here that all cell sizes are the same
        for i, dim in enumerate(DimensionPair):
            if i >= cells.dimensionality:
                break
            dir_L = dim.value[0].value
            dir_R = dim.value[1].value
            neigh_L = self.cells.neighbours[dir_L]
            neigh_R = self.cells.neighbours[dir_R]

            state_cls = cells._values.__class__
            slope = self.limiter(
                self.cells.values.view(state_cls).get_primitive()  # type: ignore
                - neigh_L.values.view(state_cls).get_primitive(),  # type: ignore
                neigh_R.values.view(state_cls).get_primitive()  # type: ignore
                - self.cells.values.view(state_cls).get_primitive(),  # type: ignore
            )

            self.slopes[..., dir_R] = slope
            self.slopes[..., dir_L] = -slope

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
            state_cls = cells._values.__class__

            self.cells.values_face[
                ...,
                [direction],
                [state_cls.prim_state._subset_fields_map],  # type: ignore
            ] = (
                self.cells.values.view(state_cls).get_primitive()  # type: ignore
                + 0.5 * self.slopes[..., direction]
            )

    def apply_fluxes(self, cells: MeshCellSet, dt: float):
        mcells = MUSCLMeshCellSet(cells)
        mcells.values -= np.einsum(  # type: ignore
            "...kl,...->...kl", self._fluxes, dt / cells.volumes
        )

    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet):
        # Solve the Riemann problem to compute the intercell flux
        # using the extrapolated and half-timestep updated states of each
        # interface as initial conditions.

        direction = neighs.direction
        oppDirection = direction + 1 if direction % 2 == 0 else direction - 1
        Q_L = self.cells.values_face[..., [direction], :]
        Q_R = self.cells.neighbours[direction].values_face[..., [oppDirection], :]
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

        self.cells = MUSCLMeshCellSet(cells)

        state_cls = cells._values.__class__
        self.slopes = np.empty(
            self.cells.values.view(state_cls).get_primitive().shape  # type: ignore
            + (2**cells.dimensionality,)
        ).view(cells._values.__class__)

    def pre_accumulate(self, cells: MeshCellSet, dt: float, t: float):
        super().pre_accumulate(cells, dt, t)

        self.slopes.fill(0)
        self.cells = MUSCLMeshCellSet(cells)

        # Initialize state values at each face with the state value
        # of the cell
        for dir in range(2**cells.dimensionality):
            self.cells._values_face[..., [dir], :] = self.cells._values.copy()

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
            self.post_extrapolation(self.cells.values_face[..., dir, :])


class MUSCL_Hancock(MUSCL):
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

            Q_L = self.cells.values_face[..., [dir_L], :]
            Q_R = self.cells.values_face[..., [dir_R], :]
            F_L = np.einsum(
                "...mkl,...l->...mk",
                self.problem.F(Q_L),
                cells.neighbours[dir_L].normals,
            )
            F_R = np.einsum(
                "...mkl,...l->...mk",
                self.problem.F(Q_R),
                cells.neighbours[dir_R].normals,
            )

            cons_states = cells._values.__class__.cons_state  # type: ignore
            cons_fields = cons_states._subset_fields_map  # type: ignore
            # BUG: the use of the brackts slice [dir_K] [cons_fields] is not
            # advised here and does not update the values_face
            self.cells.values_face[..., [dir_L], [cons_fields]] -= (  # type: ignore
                0.5
                * dt
                / cells.volumes[
                    ...,
                    np.newaxis,
                    np.newaxis,
                ]
                * cells.surfaces[..., np.newaxis, [dir_L]]
                * (F_L + F_R)
            )
            self.cells.values_face[..., [dir_R], [cons_fields]] -= (  # type: ignore
                0.5
                * dt
                / cells.volumes[
                    ...,
                    np.newaxis,
                    np.newaxis,
                ]
                * cells.surfaces[..., np.newaxis, [dir_R]]
                * (F_L + F_R)
            )

    def pre_accumulate(self, cells: MeshCellSet, dt: float, t: float):
        super().pre_accumulate(cells, dt, t)

        # Perform the half-timestep at each interface using cell
        # flux (for the conserved components)
        self.update_values_face(cells, dt)

        # Update the auxiliary components at each face
        for dir in range(2**cells.dimensionality):
            self.post_extrapolation(self.cells.values_face[..., dir, :])
