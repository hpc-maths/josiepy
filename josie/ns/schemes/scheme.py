# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from josie.euler.schemes import EulerScheme
from josie.ns.problem import NSProblem
from josie.scheme.diffusive import DiffusiveScheme

if TYPE_CHECKING:
    from josie.euler.eos import EOS
    from josie.ns.transport import NSTransport


class NSScheme(EulerScheme, DiffusiveScheme):
    problem: NSProblem

    def __init__(self, eos: EOS, transport: NSTransport):
        self.problem = NSProblem(eos, transport)
