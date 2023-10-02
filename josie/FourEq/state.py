# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

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


class FourEqPrimFields(Fields):
    """Indexing fields for a substate associated to a phase"""

    alpha = 0
    rho = 1
    P = 2
    U = 3
    V = 4


class FourEqPhaseFields(Fields):
    """Indexing fields for a substate associated to a phase"""

    arho = 0
    p = 1
    c = 2


class FourEqPhaseState(PhaseState):
    """State array for one single phase"""

    fields = FourEqPhaseFields
    full_state_fields = FourEqFields


class FourEqPrimState(SubsetState):
    """State array for one single phase"""

    fields = FourEqPrimFields
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
    prim_state = FourEqPrimState
    phase_state = FourEqPhaseState

    def get_conservative(self) -> FourEqConsState:
        return super().get_conservative().view(FourEqConsState)

    def get_phase(self, phase: Phases) -> FourEqPhaseState:
        return super().get_phase(phase).view(FourEqPhaseState)
