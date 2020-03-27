# josiepy
# Copyright © 2019 Ruben Di Battista
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Ruben Di Battista ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Ruben Di Battista BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Ruben Di Battista.
from __future__ import annotations

import numpy as np

from enum import IntEnum
from typing import Any, Dict

from josie.solver.state import State


class PhasePair:
    """ A tuple of objects that are indexable by :class:`Phases`.
    """

    def __init__(self, phase1: Any, phase2: Any):
        self._data: Dict[Phases, int] = {
            Phases.PHASE1: phase1,
            Phases.PHASE2: phase2,
        }

    def __getitem__(self, phase: Phases):
        return self._data[phase]

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    @property
    def phase1(self) -> Any:
        return self._data[Phases.PHASE1]

    @property
    def phase2(self) -> Any:
        return self._data[Phases.PHASE2]


class Phases(IntEnum):
    """ A phase indicator :class:`IntEnum`. It gives the index within the
    :class:`Q` array where that phase state variables begin

    """

    PHASE1 = 1
    PHASE2 = 10


class Fields(IntEnum):
    alpha = 0

    arho1 = 1
    arhoU1 = 2
    arhoV1 = 3
    arhoE1 = 4
    rhoe1 = 5
    U1 = 6
    V1 = 7
    p1 = 8
    c1 = 9

    arho2 = 10
    arhoU2 = 11
    arhoV2 = 12
    arhoE2 = 13
    arhoe2 = 14
    U2 = 15
    V2 = 16
    p2 = 17
    c2 = 18


class ConsFields(IntEnum):
    arho1 = 0
    arhoU1 = 1
    arhoV1 = 2
    arhoE1 = 3

    arho2 = 4
    arhoU2 = 5
    arhoV2 = 6
    arhoE2 = 7


class FluxFields(IntEnum):
    alpha = 0
    arho1 = 1
    arhoU1 = 2
    arhoV1 = 3
    arhoE1 = 4

    arho2 = 5
    arhoU2 = 6
    arhoV2 = 7
    arhoE2 = 8


class PhaseFields(IntEnum):
    """ Indexing fields for a substate associated to a phase """

    arho = 0
    arhoU = 1
    arhoV = 2
    arhoE = 3
    rhoe = 4
    U = 5
    V = 6
    p = 7
    c = 8


class ConsPhaseFields(IntEnum):
    """ Indexing fields for the conservative part of a phase state"""

    arho = 0
    arhoU = 1
    arhoV = 2
    arhoE = 3


class ConsPhaseQ(State):
    """ State array for the conservative part of a phase state """

    fields = ConsPhaseFields


class PhaseQ(State):
    """ State array for one single phase """

    fields = PhaseFields

    def get_conservative(self) -> ConsPhaseQ:
        return self[..., self.fields.arho : self.fields.arhoE + 1].view(
            ConsPhaseQ
        )

    def set_conservative(self, values: ConsPhaseQ):
        self[..., self.fields.arho : self.fields.arhoE + 1] = values


class ConsQ(State):
    """ State array for conservtive part of the state of one single phase """

    fields = ConsFields


class FluxQ(State):
    """ State arry for the convective flux"""

    fields = FluxFields


class Q(State):
    r""" We create one big state that contains the actual conservative
    variables that are used in the flux together with the "auxiliary" variables
    that are instead needed, for example, to compute the speed of sound.

    The state of system described in :cite:`baer_nunziato` is actually two
    Euler states togeter with the state associated to the volume fraction
    :math:`\alpha` """

    fields = Fields

    def get_phase(self, phase: Phases) -> PhaseQ:
        r""" Returns the part of the state associated to a specified ``phase``
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

        return self[..., phase : phase + 9].view(PhaseQ)

    def set_phase(self, phase: Phases, values: PhaseQ):
        """ Sets the part of the system associated to the specified ``phase``
        with the provided `values`

        Parameters
        ---------
        phase
            A :class:`Phases` instance identifying the phase
            partition of the state for which the values need to be set

        values
            The corresponding values to update the state with
        """

        self[..., phase : phase + 9] = values

    def get_conservative(self) -> ConsQ:
        """ Returns the conservative part of the state """
        fields = self.fields
        indices = np.array(
            [
                fields.arho1,
                fields.arhoU1,
                fields.arhoV1,
                fields.arhoE1,
                fields.arho2,
                fields.arhoU2,
                fields.arhoV2,
                fields.arhoE2,
            ]
        )

        return self[..., indices].view(ConsQ)

    def set_conservative(self, values: ConsQ):
        fields = self.fields
        indices = np.array(
            [
                fields.arho1,
                fields.arhoU1,
                fields.arhoV1,
                fields.arhoE1,
                fields.arho2,
                fields.arhoU2,
                fields.arhoV2,
                fields.arhoE2,
            ]
        )

        self[..., indices] = values

    def set_phase_conservative(self, phase: Phases, values: ConsPhaseQ):
        """ Sets the conservative part of the state of the given ``phase`` """

        self[..., phase : phase + 4] = values
