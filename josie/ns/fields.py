# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" Navier-Stokes Equations take much of the implementations from :mod:`euler`
"""
from __future__ import annotations

from josie.fields import Fields


class NSGradientFields(Fields):
    U = 0
    V = 1
    rhoe = 2
