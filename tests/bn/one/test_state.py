import numpy as np
import pytest

from josie.twofluid.state import PhasePair, Phases
from josie.bn.state import Q


@pytest.fixture
def state():
    yield np.array(range(len(Q.fields))).view(Q)


def test_phase_pair():
    pair = PhasePair(3, 4)

    assert pair[Phases.PHASE1] == 3
    assert pair[Phases.PHASE2] == 4


def test_phase(state):
    fields = Q.fields
    assert np.array_equal(
        state.get_phase(Phases.PHASE1),
        state[
            ...,
            [
                fields.arho1,
                fields.arhoU1,
                fields.arhoV1,
                fields.arhoE1,
                fields.rhoe1,
                fields.U1,
                fields.V1,
                fields.p1,
                fields.c1,
            ],
        ],
    )

    assert np.array_equal(
        state.get_phase(Phases.PHASE2),
        state[
            ...,
            [
                fields.arho2,
                fields.arhoU2,
                fields.arhoV2,
                fields.arhoE2,
                fields.rhoe2,
                fields.U2,
                fields.V2,
                fields.p2,
                fields.c2,
            ],
        ],
    )


def test_conservative(state):
    fields = Q.fields
    assert np.array_equal(
        state.get_conservative(),
        state[
            ...,
            [
                fields.alpha,
                fields.arho1,
                fields.arhoU1,
                fields.arhoV1,
                fields.arhoE1,
                fields.arho2,
                fields.arhoU2,
                fields.arhoV2,
                fields.arhoE2,
            ],
        ],
    )
