# josiepy
# Copyright © 2021 Ruben Di Battista
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
