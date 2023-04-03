# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.fields import Fields
from josie.state import State


class HeatFields(Fields):
    T = 0


class Q(State):
    fields = HeatFields
