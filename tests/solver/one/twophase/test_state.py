import numpy as np
import pytest

from josie.solver.twophase.state import Q, Phases, PhasePair


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
                fields.rho1,
                fields.rhoU1,
                fields.rhoV1,
                fields.rhoE1,
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
                fields.rho2,
                fields.rhoU2,
                fields.rhoV2,
                fields.rhoE2,
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
        state.get_phase(Phases.PHASE1).get_conservative(),
        state[..., [fields.rho1, fields.rhoU1, fields.rhoV1, fields.rhoE1]],
    )

    assert np.array_equal(
        state.get_phase(Phases.PHASE2).get_conservative(),
        state[..., [fields.rho2, fields.rhoU2, fields.rhoV2, fields.rhoE2]],
    )
