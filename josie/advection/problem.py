# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.scheme.convective import ConvectiveProblem

from josie.advection.state import Q


class AdvectionProblem(ConvectiveProblem):
    V: np.ndarray

    def __init__(self, V: np.ndarray):
        self.V = V

    def flux(self, state_array: Q) -> np.ndarray:
        return np.einsum("j,...i->...ij", self.V, state_array)

    def F(self, state_array: Q) -> np.ndarray:
        # I multiply each element of the given state array by the velocity
        # vector. I obtain an Nx2 array where each row is the flux on each
        # cell
        return self.flux(state_array)
