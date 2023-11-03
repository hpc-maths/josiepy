# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.fluid.fields import FluidFields
from josie.twofluid.fields import Phases
from josie.twofluid.state import TwoFluidState, PhaseState

from josie.fields import Fields
from josie.state import SubsetState


class TsCapFields(FluidFields):
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
    capSigma = 17

    grada_x = 18
    grada_y = 19
    n_x = 20
    n_y = 21
    norm_grada = 22
    H = 23

    MaX = 24
    MaY = 25
    WeX = 26
    WeY = 27
    c_cap1X = 28
    c_cap1Y = 29
    c_cap2X = 30
    c_cap2Y = 31


class TsCapConsFields(Fields):
    """Indexing fields for the consevative part of the full state"""

    abarrho = 0
    rhoU = 1
    rhoV = 2

    arho1 = 3
    arho2 = 4
    arho1d = 5
    ad = 6
    capSigma = 7


class TsCapPrimFields(Fields):
    """Indexing fields for the consevative part of the full state"""

    rho = 0
    U = 1
    V = 2

    abar = 3
    arho1d = 4
    ad = 5
    capSigma = 6


class TsCapPhaseFields(Fields):
    """Indexing fields for a substate associated to a phase"""

    arho = 0
    p = 1
    c = 2


class TsCapPhaseState(PhaseState):
    """State array for one single phase"""

    fields = TsCapPhaseFields
    full_state_fields = TsCapFields


class TsCapConsState(SubsetState):
    """State array for conservative part of the state"""

    fields = TsCapConsFields
    full_state_fields = TsCapFields


class TsCapPrimState(SubsetState):
    """State array for conservative part of the state"""

    fields = TsCapPrimFields
    full_state_fields = TsCapFields


class Q(TwoFluidState):
    r"""We create one big state that contains the actual conservative
    variables that are used in the flux together with the "auxiliary" variables
    that are instead needed, for example, to compute the speed of sound."""

    fields = TsCapFields
    cons_state = TsCapConsState
    prim_state = TsCapConsState
    phase_state = TsCapPhaseState

    def get_conservative(self) -> TsCapConsState:
        return super().get_conservative().view(TsCapConsState)

    def get_phase(self, phase: Phases) -> TsCapPhaseState:
        return super().get_phase(phase).view(TsCapPhaseState)
