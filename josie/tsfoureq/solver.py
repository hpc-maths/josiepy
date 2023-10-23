# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.solver import Solver
from josie.mesh import Mesh

from .schemes import TSFourEqScheme
from .state import Q


class TSFourEqSolver(Solver):
    """A solver for the TwoPhase system"""

    def __init__(self, mesh: Mesh, scheme: TSFourEqScheme):
        super().__init__(mesh, Q, scheme)
