# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cellset import MeshCellSet, NeighboursCellSet
from josie.mesh.mesh import Mesh
from josie.state import State

from josie.scheme.convective import ConvectiveScheme


class Godunov(ConvectiveScheme):
    def post_init(self, mesh: Mesh):
        r"""Initialize the datastructure holding the values at interface
        for each cell and face
        """

        super().post_init(mesh)

        self._fluxes: State = np.empty_like(mesh.cells.values)

    def F(self, cells: MeshCellSet, neighs: NeighboursCellSet):
        # Solve the Riemann problem to compute the intercell flux
        # using cell states as initial conditions.
        Q_L = cells.values
        Q_R = neighs.values

        return self.intercellFlux(
            Q_L,
            Q_R,
            neighs.normals,
            neighs.surfaces,
        )
