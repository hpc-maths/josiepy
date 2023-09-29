# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


from typing import Type

from josie.state import State, SubsetState

from .fields import FluidFields


class ConsState(State):
    """A mixin providing methods to retrieve the conservative part of a
    :class:`State`"""

    prim_state: Type[SubsetState]
    cons_state: Type[SubsetState]

    def get_conservative(self) -> SubsetState:
        """Returns the conservative part of the state"""
        return self[..., self.cons_state._subset_fields_map]  # type: ignore

    def set_conservative(self, values: State):
        """Set the conservative part of the state"""
        self[..., self.cons_state._subset_fields_map] = values

    def get_primitive(self) -> SubsetState:
        """Returns the diffusive part of the state"""
        return self[..., self.prim_state._subset_fields_map]  # type: ignore

    def set_primitive(self, values: State):
        """Set the diffusive part of the state"""
        self[..., self.prim_state._subset_fields_map] = values


class DiffState(State):
    """A mixin providing methods to retrieve the diffusive part of a
    :class:`State`
    """

    diff_state: Type[SubsetState]

    def get_diffusive(self) -> SubsetState:
        """Returns the diffusive part of the state"""
        return self[..., self.diff_state._subset_fields_map]  # type: ignore

    def set_diffusive(self, values: State):
        """Set the diffusive part of the state"""
        self[..., self.diff_state._subset_fields_map] = values


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
