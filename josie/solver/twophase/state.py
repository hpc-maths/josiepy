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

from enum import IntEnum
from typing import Any, Dict

from josie.solver.state import State
from josie.solver.euler.state import Q as EulerQ


class Phases(IntEnum):
    """ A phase indicator :class:`IntEnum`. It gives the index within the
    :class:`State` array where that phase state variables begin

    """

    PHASE1 = 1
    PHASE2 = 10


class PhasePair:
    """ A tuple of objects that are indexable by :class:`Phases`.
    """

    # Remap the phases int values to the right one for this tuple
    _index: Dict[Phases, int] = {Phases.PHASE1: 0, Phases.PHASE2: 1}

    def __init__(self, obj1: Any, obj2: Any):
        self._data = (obj1, obj2)

    def __getitem__(self, phase: Phases):
        return self._data[self._index[phase]]


class Fields(IntEnum):
    alpha = 0

    rho1 = 1
    rhoU1 = 2
    rhoV1 = 3
    rhoE1 = 4
    rhoe1 = 5
    U1 = 6
    V1 = 7
    p1 = 8
    c1 = 9

    rho2 = 10
    rhoU2 = 11
    rhoV2 = 12
    rhoE2 = 13
    rhoe2 = 14
    U2 = 15
    V2 = 16
    p2 = 17
    c2 = 18


class Q(State):
    r""" We create one big state that contains the actual conservative
    variables that are used in the flux together with the "auxiliary" variables
    that are instead needed, for example, to compute the speed of sound.

    The state of system described in :cite:`baer_nunziato` is actually two
    Euler states togeter with the state associated to the volume fraction
    :math:`\alpha` """

    fields = Fields

    def get_phase(self, phase: Phases) -> EulerQ:
        r""" Returns the part of the state associated to a specified phase
        as an instance of :class:`~euler.state.Q`

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

        return self[..., phase : phase + 9].view(EulerQ)

    def set_phase(self, phase: Phases, values: EulerQ):
        """ Sets the part of the system associated to the given `phase` with
        the provided `values`

        Parameters
        ---------
        phase
            A :class:`Phases` instance identifying the phase
            partition of the state for which the values need to be set

        values
            The corresponding values to update the state with
        """

        self[..., phase : phase + 9] = values
