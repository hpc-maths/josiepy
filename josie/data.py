# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    # This is a trick to enable mypy to evaluate the Enum as a standard
    # library Enum for type checking but we use `aenum` in the running code
    from enum import Enum, IntEnum  # pragma: no cover

    NoAlias = object()  # pragma: no cover
else:
    from aenum import Enum, IntEnum, NoAlias

StateData = Dict[str, np.ndarray]


@dataclass
class StateElement:
    """An handy class to store the state of a simulation"""

    time: float
    data: StateData


def unpickle_noaliasenum(cls, name):
    return getattr(cls, name)


class NoAliasEnum(Enum, settings=NoAlias):
    """Allow pickling of NoAlias IntEnum"""

    def __reduce_ex__(self, proto):
        return (unpickle_noaliasenum, (self.__class__, self.name))


class NoAliasIntEnum(IntEnum, settings=NoAlias):
    """Allow pickling of NoAlias IntEnum"""

    def __reduce_ex__(self, proto):
        return (unpickle_noaliasenum, (self.__class__, self.name))
