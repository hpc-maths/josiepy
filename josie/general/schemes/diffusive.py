import numpy as np

from typing import Sequence

from josie.mesh.cellset import MeshCellSet, CellSet
from josie.solver.scheme.diffusive import DiffusiveScheme


class LeastSquareGradient(DiffusiveScheme):
    r""" A mixin that provides the approximation of the gradient term in the
    diffusive term as a least square approximation over all the neighbours of
    the mesh cells

    Given a neighbour cell :math:`N` and a generic cell :math:`C`, the value of
    a generic field :math:`\phi_N` can be approximated as:

    .. math::

        \phi_N = \phi_C + \nabla \phi_C \cdot \qty(\vb{r}_N - \vb{r}_C)


    So we can optimize a cost function

    .. math::

        G_C = \sum_\text{neigh} \phi_C + \nabla \phi_C \cdot \qty(\vb{r}_N -
            \vb{r}_C) - \phi_N

    to obtain the value of the gradient in the cell :math:`\nabla \phi_C`
    """

    def post_init(self, cells: MeshCellSet, neighbours: Sequence[CellSet]):
        r""" Initialize the datastructure holding the matrix used to solve
        the Least Square problem and also the RHS of the linear system

        """

        nx, ny, num_points, dimensionality = cells.centroids.shape

        super().post_init(cells, neighbours)

        self._A = np.zeros((nx, ny, dimensionality, dimensionality))
        self._RHS = np.zeros_like(self._gradient)

        # Store the relative vectors between cells and neighbour per each set
        # of neighbour. There are 2*dimensionality neighbours (i.e. 2D -> 4
        # neighbours)
        self._r = np.zeros(
            (nx, ny, num_points, 2 * dimensionality, dimensionality)
        )

        # Store norm of the relative vectors to be used as weights
        self._w = np.zeros((nx, ny, num_points, 2 * dimensionality))

        # If the mesh is static, the A matrix does not change. So we can
        # initialize it once and for all here

        # Pre-allocate A
        A = np.zeros_like(self._A)

        for i, neigh in enumerate(neighbours):
            # Compute relative vector between cells and neighbour
            r = neigh.centroids - cells.centroids

            # Compute unweighted A components for this neighbour
            A = np.einsum("...ki,...kj->...ij", r, r)

            # Weight by the inverse of the relative vector norm squared. Ie
            # the trace of each A matrix per each cell
            w = (
                1
                / np.trace(A, axis1=-1, axis2=-2)[..., np.newaxis, np.newaxis]
            )

            # Add to global A
            self._A += A * w

            # Store relative vector
            self._r[..., i, :] = r

            # ... and weight (keeping shape)
            self._w[..., i, np.newaxis] = w

    def pre_step(self, cells: MeshCellSet, neighbours: Sequence[CellSet]):

        for i, neigh in enumerate(neighbours):
            r = self._r[..., i, :]
            w = self._w[..., i, np.newaxis]

            r *= w

            # Compute RHS
            self._RHS += np.einsum(
                "...i,...kj->...ij", neighbours[i].values - cells.values, r
            )

        self._gradient = np.linalg.solve(
            self._A[..., np.newaxis, :, :], self._RHS
        )
