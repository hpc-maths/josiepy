# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cellset import MeshCellSet
from josie.scheme.diffusive import DiffusiveScheme


class LeastSquareGradient(DiffusiveScheme):
    r"""The gradient term in the diffusive term is approximated as a least
    square approximation over all the neighbours of the mesh cells

    Given a neighbour cell :math:`N` and a generic cell :math:`C`, the value of
    a generic field :math:`\phi_N` can be approximated as:

    .. math::

        \phi_N = \phi_C + \nabla \phi_C \cdot \qty(\vb{r}_N - \vb{r}_C)


    So we can optimize a cost function

    .. math::

        G_C = \sum_\text{neigh} \phi_C + \nabla \phi_C \cdot \qty(\vb{r}_N -
            \vb{r}_C) - \phi_N

    to obtain the value of the gradient in the cell :math:`\nabla \phi_C`

    :cite:`moukalled_finite_2016`
    """

    _gradient: np.ndarray

    def _init_gradient(self, cells: MeshCellSet):
        r"""Initialize the datastructure holding the gradient
        :math:`\pdeGradient, \ipdeGradient` per each cell
        """

        nx, ny, num_dofs, num_fields = cells.values.shape
        dimensionality = cells.dimensionality

        super().post_init(cells)

        self._gradient = np.zeros((nx, ny, num_dofs, num_fields, dimensionality))

    def pre_step(self, cells: MeshCellSet, dt: float):
        super().pre_step(cells, dt)

        self._gradient.fill(0)

    def post_init(self, cells: MeshCellSet):
        r"""Initialize the datastructure holding the matrix used to solve
        the Least Square problem and also the RHS of the linear system

        """

        nx, ny, num_dofs, num_fields = cells.values.shape
        dimensionality = cells.dimensionality

        super().post_init(cells)

        self._init_gradient(cells)

        self._A = np.zeros((nx, ny, dimensionality, dimensionality))
        self._RHS = np.zeros_like(self._gradient)

        # Store the relative vectors between cells and neighbour per each set
        # of neighbour. There are 2*dimensionality neighbours (i.e. 2D -> 4
        # neighbours)
        self._r = np.zeros((nx, ny, num_fields, 2 * dimensionality, dimensionality))

        # Store norm of the relative vectors to be used as weights
        self._w = np.zeros((nx, ny, num_fields, 2 * dimensionality))

        # If the mesh is static, the A matrix does not change. So we can
        # initialize it once and for all here

        # Pre-allocate A
        A = np.zeros_like(self._A)

        # TODO: Store the neighbours in a numpy array instead of a list. That's
        # gonna be faster
        for i, neigh in enumerate(cells.neighbours):
            # Compute relative vector between cells and neighbour
            # using only the components associated to the problem
            # dimensionality (i.e. 1D -> first component, 2D -> first two)
            r = (
                neigh.centroids[..., :dimensionality]
                - cells.centroids[..., :dimensionality]
            )

            # Compute unweighted A components for this set of neighbours
            A = np.einsum("...ki,...kj->...ij", r, r)

            # Weight by the inverse of the relative vector norm squared. Ie
            # the trace of each A matrix per each cell
            w = 1 / np.trace(A, axis1=-1, axis2=-2)[..., np.newaxis, np.newaxis]

            # Add to global A
            self._A += A * w

            # Store relative vector
            self._r[..., i, :] = r

            # ... and weight (keeping shape)
            self._w[..., i, np.newaxis] = w

    def pre_accumulate(self, cells: MeshCellSet, dt: float, t: float):
        for i, neigh in enumerate(cells.neighbours):
            r = self._r[..., i, :]
            w = self._w[..., i, np.newaxis]

            r *= w

            # Compute RHS
            self._RHS += np.einsum(
                "...mi,...kj->...mij",
                cells.neighbours[i].values - cells.values,
                r,
            )

        # The np.newaxis are needed to allow correct broadcasting against RHS
        # TODO: make less ugly?
        self._gradient = np.linalg.solve(
            self._A[..., np.newaxis, np.newaxis, :, :], self._RHS
        )
