from __future__ import annotations

import numpy as np


from typing import Type

from josie.state import State

from .fields import FluidFields


class FluidState(State):
    """A class used for type checking to indicate a state for a fluid dynamics
    problem, i.e. a state whose fields are :class:`FluidFields`, that is they
    have velocity components. This class also allows to retrieve the
    conservative part of the state

    Attributes
    ----------
    fields
        The indexing :class:`FluidFields` for all the variables

    cons_state
        An associated :class:`State` that wraps the conservative subset of
        fields
    """

    fields: Type[FluidFields]
    cons_state: Type[State]

    def __init_subclass__(cls):
        super.__init_subclass__()
        cls._cons_fields = np.array(
            [
                field
                for field in cls.fields
                if field.name in cls.cons_state.fields.names()
            ]
        )

    def get_conservative(self) -> State:
        """ Returns the conservative part of the state """
        return self[..., self._cons_fields]

    def set_conservative(self, values: State):
        self[..., self._cons_fields] = values
