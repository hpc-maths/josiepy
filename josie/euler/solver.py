# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from josie.solver import Solver

from .state import EulerState
from .schemes import EulerScheme

if TYPE_CHECKING:
    from josie.mesh import Mesh


class EulerSolver(Solver):
    """A solver for the Euler system"""

    def __init__(self, mesh: Mesh, scheme: EulerScheme):

        super().__init__(mesh, EulerState, scheme)
