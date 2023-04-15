# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from josie.solver import Solver

from .state import NSState
from .schemes import NSScheme

if TYPE_CHECKING:
    from josie.mesh import Mesh


class NSSolver(Solver):
    """A solver for the Euler system"""

    def __init__(self, mesh: Mesh, scheme: NSScheme):
        super().__init__(mesh, NSState, scheme)
