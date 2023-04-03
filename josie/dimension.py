# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import IntEnum


class Dimensionality(IntEnum):
    ZEROD = 0
    ONED = 1
    TWOD = 2
    THREED = 3


# TODO: Change when going to 3D
MAX_DIMENSIONALITY = Dimensionality.TWOD
