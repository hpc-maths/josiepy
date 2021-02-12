from __future__ import annotations


from typing import Type

from josie.state import State, SubsetState

from .fields import FluidFields


class ConsSubsetState(SubsetState, abstract=True):
    """A :class:`SubsetState` holding the conservative subset of a
    system"""

    pass


class ConsState(State):
    """A mixin providing methods to retrieve the conservative part of a
    :class:`State`"""

    cons_state: Type[ConsSubsetState]

    def get_conservative(self) -> State:
        """ Returns the conservative part of the state """
        return self[..., self.cons_state._subset_fields_map]

    def set_conservative(self, values: State):
        """ Set the conservative part of the state """
        self[..., self.cons_state._subset_fields_map] = values


class SingleFluidState(ConsState):
    """A class used for type checking to indicate a state for a fluid dynamics
    problem, i.e. a state whose fields are :class:`FluidFields`, that is they
    have velocity components.

    Attributes
    ----------
    fields
        The indexing :class:`FluidFields` for all the variables

    cons_state
        An associated :class:`State` that wraps the conservative subset of
        fields
    """

    fields: Type[FluidFields]
