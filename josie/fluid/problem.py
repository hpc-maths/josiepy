# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.problem import Problem
from josie.transport import Transport


class DiffusiveProblem(Problem):
    """A :class:`Problem` providing a :class:`Transport` attribute to compute
    transport coefficients"""

    def __init__(self, transport: Transport, **kwargs):
        super().__init__(**kwargs)

        self.transport = transport
