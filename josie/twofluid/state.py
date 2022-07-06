from __future__ import annotations

import re

from typing import Any, Optional, Type

from josie.fluid.state import ConsState
from josie.state import SubsetState
from josie.twofluid import fields


class PhasePair(dict):
    """A tuple of objects that are indexable by :class:`Phases`."""

    def __init__(
        self, phase1: Optional[Any] = None, phase2: Optional[Any] = None
    ):
        _dict = {
            fields.Phases.PHASE1: phase1,
            fields.Phases.PHASE2: phase2,
        }

        super().__init__(_dict)

    def __repr__(self):

        key1 = fields.Phases.PHASE1
        key2 = fields.Phases.PHASE2

        val1 = self[key1]
        val2 = self[key2]

        return f"{{ {key1.name}:{val1}, {key2.name}:{val2}  }}"

    @property
    def phase1(self) -> Any:
        return super().__getitem__(fields.Phases.PHASE1)

    @property
    def phase2(self) -> Any:
        return super().__getitem__(fields.Phases.PHASE1)


class PhaseState(SubsetState, abstract=True):
    """This :class:`ConsSubsetState` stores two sets of indices, one per
    phase, and it matches the field names with a regex in order to detect the
    phase they are member of. This is needed to support out-of-order listing of
    fields"""

    _REGEX_PATTERN = re.compile(r"\w+(\d{1})")
    _subset_fields_map: PhasePair

    def __init_subclass__(cls, /, abstract=False, **kwargs):
        cls._subset_fields_map = PhasePair(phase1=[], phase2=[])
        if not (abstract):
            phase_fields = {fields.Phases.PHASE1: [], fields.Phases.PHASE2: []}

            # Add the matched fields to the corresponding entry in the
            # phase_fields dictionary
            for field in cls.full_state_fields:
                if match := cls._REGEX_PATTERN.match(field.name):
                    phase_number = int(match.group(1))

                    # TODO: If more than 2 phases needed, this needs to be
                    # generalized
                    if phase_number == 1:
                        phase_fields[fields.Phases.PHASE1].append(field)
                    else:
                        phase_fields[fields.Phases.PHASE2].append(field)

            cls._subset_fields_map.update(phase_fields)


class TwoFluidState(ConsState):
    """A generic :class:`ConsState` for a two-phase flow problem. It allows
    to retrieve individual phase states"""

    phase_state: Type[PhaseState]

    def get_phase(self, phase: fields.Phases) -> PhaseState:
        r"""Returns the part of the state associated to a specified ``phase``
        as an instance of :class:`~PhaseQ`

        .. warning::

            This does not return the first variable of the state, i.e.
            :math:`\alpha`

        Parameters
        ---------
        phase
            A :class:`Phases` instance identifying the requested phase
            partition of the state

        Returns
        ------
        state
            A :class:`~euler.state.Q` instance corresponding to the partition
            of the system associated to the requested phase
        """

        return self[..., self.phase_state._subset_fields_map[phase]].view(
            PhaseState
        )

    def set_phase(self, phase: fields.Phases, values: PhaseState):
        """Sets the part of the system associated to the specified ``phase``
        with the provided `values`

        Parameters
        ---------
        phase
            A :class:`Phases` instance identifying the phase
            partition of the state for which the values need to be set

        values
            The corresponding values to update the state with
        """

        self[..., self.phase_state._subset_fields_map[phase]] = values
