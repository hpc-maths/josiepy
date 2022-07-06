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

from josie.fluid.fields import FluidFields
from josie.twofluid.fields import Phases
from josie.twofluid.state import TwoFluidState, PhaseState

from josie.fields import Fields
from josie.state import SubsetState


class FourEqFields(FluidFields):

    arho = 0
    rhoU = 1
    rhoV = 2
    rho = 3
    U = 4
    V = 5
    P = 6
    c = 7

    alpha = 8

    arho1 = 9
    p1 = 10
    c1 = 11

    arho2 = 12
    p2 = 13
    c2 = 14


class FourEqConsFields(Fields):
    """Indexing fields for the consevative part of the full state"""

    arho = 0
    rhoU = 1
    rhoV = 2

    arho1 = 3
    arho2 = 4


class FourEqPhaseFields(Fields):
    """Indexing fields for a substate associated to a phase"""

    arho = 0
    p = 1
    c = 2


class FourEqPhaseState(PhaseState):
    """State array for one single phase"""

    fields = FourEqPhaseFields
    full_state_fields = FourEqFields


class FourEqConsState(SubsetState):
    """State array for conservative part of the state"""

    fields = FourEqConsFields
    full_state_fields = FourEqFields


class Q(TwoFluidState):
    r"""We create one big state that contains the actual conservative
    variables that are used in the flux together with the "auxiliary" variables
    that are instead needed, for example, to compute the speed of sound."""

    fields = FourEqFields
    cons_state = FourEqConsState
    phase_state = FourEqPhaseState

    def get_conservative(self) -> FourEqConsState:
        return super().get_conservative().view(FourEqConsState)

    def get_phase(self, phase: Phases) -> FourEqPhaseState:
        return super().get_phase(phase).view(FourEqPhaseState)
