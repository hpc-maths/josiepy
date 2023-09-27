# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


from josie.scheme.diffusive import DiffusiveScheme
from josie.mesh.cellset import NeighboursCellSet, MeshCellSet


class CentralDifferenceGradient(DiffusiveScheme):
    r"""The gradient term in the diffusive term is approximated as a simple
    directional derivative using node values of the cells and their
    corresponding neighbour

    .. math::

        \gradient{\pdeState} \cdot \pdeNormal =
        \frac{\pdeState_R - \pdeState_L}{\Delta x}
    """

    def post_init(self, cells: MeshCellSet):
        r"""Initialize the datastructure holding the norm of the relative
        vector between cells and their neighbours

        .. math::

            \delta x = \norm{\vb{r_R} - \vb{r_L}}
        """

        # TODO: Add num_dofs into the size to allow for multiple dofs in a
        # single cell
        nx, ny, _, _ = cells.values.shape
        dimensionality = cells.dimensionality
        num_neighbours = 2 * dimensionality

        super().post_init(cells)
        self._fluxes = np.empty_like(cells.values)

        # Norm of the relative distance
        self._r = np.zeros((nx, ny, num_neighbours))

        # Store a mapping between the index of the neighbour cell set and its
        # direction
        self._directions = {}

        # TODO: Store neighbours in numpy array
        for idx, neighs in enumerate(cells.neighbours):
            # Compute relative vector between cells and neighbour
            # using only the components associated to the problem
            # dimensionality (idx.e. 1D -> first component, 2D -> first two)
            r = (
                neighs.centroids[..., :dimensionality]
                - cells.centroids[..., :dimensionality]
            )

            dx = np.linalg.norm(r, axis=-1)

            # Store relative vector norm
            self._r[..., idx, np.newaxis] = dx

            # Store the position of the neighbour set, indexed by its direction
            self._directions[neighs.direction] = idx

    def D(self, cells: MeshCellSet, neighs: NeighboursCellSet):
        # Retrieve neighbour index
        idx = self._directions[neighs.direction]

        # Retrieve length of the relative vector between cell and neighbour
        r = self._r[..., idx, np.newaxis, np.newaxis]

        # Estimate the gradient in normal direction
        dQ = (neighs.values - cells.values) / r

        KdQ = np.einsum("...mijkl,...mj->...mi", self.problem.K(cells), dQ)

        # Multiply by the surface
        return KdQ * neighs.surfaces[..., np.newaxis, np.newaxis]
