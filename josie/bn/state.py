# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from josie.fluid.fields import FluidFields
from josie.twofluid.fields import Phases
from josie.twofluid.state import TwoFluidState, PhaseState

from josie.fields import Fields
from josie.state import SubsetState


class BaerFields(Fields):
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
    rhoe2 = 14
    U2 = 15
    V2 = 16
    p2 = 17
    c2 = 18


class BaerConsFields(Fields):
    """Indexing fields for the consevative part of the full state"""

    alpha = 0
    arho1 = 1
    arhoU1 = 2
    arhoV1 = 3
    arhoE1 = 4

    arho2 = 5
    arhoU2 = 6
    arhoV2 = 7
    arhoE2 = 8


class BaerPhaseFields(FluidFields):
    """Indexing fields for a substate associated to a phase"""

    arho = 0
    arhoU = 1
    arhoV = 2
    arhoE = 3
    rhoe = 4
    U = 5
    V = 6
    p = 7
    c = 8


class BaerPhaseState(PhaseState):
    """State array for one single phase"""

    fields = BaerPhaseFields
    full_state_fields = BaerFields


class BaerConsState(SubsetState):
    """State array for conservative part of the state"""

    fields = BaerConsFields
    full_state_fields = BaerFields


class BaerGradFields(Fields):
    r"""Indexes used to index the gradient pre-factor
    :math:`\pdeNonConservativeMultiplier`. Check :mod:`~twofluid.problem` for
    more information on how the multiplier is reduced in size to optimize
    the compuation"""

    alpha = 0


class Q(TwoFluidState):
    r"""We create one big state that contains the actual conservative
    variables that are used in the flux together with the "auxiliary" variables
    that are instead needed, for example, to compute the speed of sound.

    The state of system described in :cite:`baer_two-phase_1986` is actually
    two Euler states together with the state associated to the volume fraction
    :math:`\alpha`"""

    fields = BaerFields
    cons_state = BaerConsState
    phase_state = BaerPhaseState

    def get_conservative(self) -> BaerConsState:
        return super().get_conservative().view(BaerConsState)

    def get_phase(self, phase: Phases) -> BaerPhaseState:
        return super().get_phase(phase).view(BaerPhaseState)
