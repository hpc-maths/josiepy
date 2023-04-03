# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.solver import Solver
from josie.mesh import Mesh

from .schemes import BaerScheme
from .state import Q


class BaerSolver(Solver):
    """A solver for the TwoPhase system"""

    def __init__(self, mesh: Mesh, scheme: BaerScheme):
        super().__init__(mesh, Q, scheme)
