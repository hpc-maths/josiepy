# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from josie.solver import Solver

from .state import Q
from .schemes import HeatScheme

if TYPE_CHECKING:
    from josie.mesh import Mesh


class HeatSolver(Solver):
    """A solver for a system governed by the heat equation"""

    def __init__(self, mesh: Mesh, scheme: HeatScheme):
        super().__init__(mesh, Q, scheme)
