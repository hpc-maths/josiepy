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
r"""
We create one big state that contains the actual conservative variables that
are used in the flux together with the "auxiliary" variables that are instead
needed, for example, to compute the speed of sound.

The state of system described in :cite:`baer_nunziato` is actually two Euler
states togeter with the state associated to the volume fraction :math:`\alpha`
"""
from __future__ import annotations

from enum import IntEnum

from josie.solver.state import State


class Phases(IntEnum):
    """ A phase indicator :class:`IntEnum`. It gives the index within the
    :class:`State` array where that phase state variables begin

    """

    PHASE1 = 1
    PHASE2 = 10


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
    fields = Fields

    def phase(self, phase: Phases) -> Q:
        r""" Returns the part of the state associated to a specified phase

        ..warning::
        This does not return the first variable of the state, i.e.
        :math:`\alpha`

        """

        return self[..., phase : phase + 8]

    def conservative(self, phase: Phases) -> Q:
        """ Returns the conservative part of the state, for a specified phase
        """

        return self[..., phase : phase + 3]
