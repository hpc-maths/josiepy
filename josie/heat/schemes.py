# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


from josie.scheme.diffusive import DiffusiveScheme

from .problem import HeatProblem


class HeatScheme(DiffusiveScheme):
    def __init__(self, problem: HeatProblem):
        super().__init__(problem)
