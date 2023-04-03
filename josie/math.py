# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" General purpose math primitives """
import numpy as np

from enum import IntEnum

from .dimension import MAX_DIMENSIONALITY


def map01to(x, a, b):
    r"""Maps :math:`x` in :math:`[0, 1] \to [a, b]`"""

    return (b - a) * x + a


class Direction(IntEnum):
    """An :class:`Enum` encapsulating the coordinates indices"""

    X = 0
    Y = 1
    Z = 2


class R3:
    """An :class:`Enum` encapsulating the unit vectors for a cartesia R2
    space
    """

    _eye = np.eye(MAX_DIMENSIONALITY)

    X = _eye[:, Direction.X]
    Y = _eye[:, Direction.Y]
    # Z = _eye[:, Direction.Z]
