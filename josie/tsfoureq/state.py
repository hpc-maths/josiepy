# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.fluid.fields import FluidFields
from josie.twofluid.fields import Phases
from josie.twofluid.state import TwoFluidState, PhaseState

from josie.fields import Fields
from josie.state import SubsetState


class TSFourEqFields(FluidFields):
    abarrho = 0
    rhoU = 1
    rhoV = 2
    rho = 3
    U = 4
    V = 5
    pbar = 6
    cFd = 7

    abar = 8

    arho1 = 9
    p1 = 10
    c1 = 11

    arho2 = 12
    p2 = 13
    c2 = 14

    arho1d = 15
    ad = 16


class TSFourEqConsFields(Fields):
    """Indexing fields for the consevative part of the full state"""

    abarrho = 0
    rhoU = 1
    rhoV = 2

    arho1 = 3
    arho2 = 4
    arho1d = 5
    ad = 6


class TSFourEqPhaseFields(Fields):
    """Indexing fields for a substate associated to a phase"""

    arho = 0
    p = 1
    c = 2


class TSFourEqPhaseState(PhaseState):
    """State array for one single phase"""

    fields = TSFourEqPhaseFields
    full_state_fields = TSFourEqFields


class TSFourEqConsState(SubsetState):
    """State array for conservative part of the state"""

    fields = TSFourEqConsFields
    full_state_fields = TSFourEqFields


class Q(TwoFluidState):
    r"""We create one big state that contains the actual conservative
    variables that are used in the flux together with the "auxiliary" variables
    that are instead needed, for example, to compute the speed of sound."""

    fields = TSFourEqFields
    cons_state = TSFourEqConsState
    prim_state = TSFourEqConsState
    phase_state = TSFourEqPhaseState

    def get_conservative(self) -> TSFourEqConsState:
        return super().get_conservative().view(TSFourEqConsState)

    def get_phase(self, phase: Phases) -> TSFourEqPhaseState:
        return super().get_phase(phase).view(TSFourEqPhaseState)
