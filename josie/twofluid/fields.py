from __future__ import annotations
from josie.fields import Fields


# FIXME: The attributes of this class are detected by mypy as `int` instead of
# `Field`. In PhasePair I needed to force the type check to int, I'd like to
# have the real thing
class Phases(Fields):
    """A phase indicator :class:`IntEnum`. It gives the index within the
    :class:`TwoFluidState` array where that phase state variables begin

    """

    PHASE1 = 1
    PHASE2 = 10
