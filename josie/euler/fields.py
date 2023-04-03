# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.fluid.fields import FluidFields
from josie.state import Fields


class ConsFields(Fields):
    """Indexing enum for the conservative state variables of the problem"""

    rho = 0
    rhoU = 1
    rhoV = 2
    rhoE = 3


class EulerFields(FluidFields):
    """Indexing enum for the state variables of the problem"""

    rho = 0
    rhoU = 1
    rhoV = 2
    rhoE = 3
    rhoe = 4
    U = 5
    V = 6
    p = 7
    c = 8
    e = 9
