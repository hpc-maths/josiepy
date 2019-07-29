import numpy as np
import pytest

from josie.solver.state import State, StateTemplate


def state_from_template():
    Q = StateTemplate('rhoU', 'rhoV', 'p')

    return Q(0, 0, 0)


def state_directly():
    return State(rhoU=0, rhoV=0, p=0)


@pytest.fixture(params=[state_from_template, state_directly])
def Q(request):
    yield request.param()


def test_wrong_number_of_fields():
    Q = StateTemplate('rhoU', 'rhoV', 'p')
    with pytest.raises(ValueError):
        Q(0, 0, 0, 0)


def test_get_attributes(Q):
    assert Q.rhoU == 0
    assert Q.rhoV == 0
    assert Q.p == 0


def test_set_attributes(Q):
    Q.p = 1.5

    assert Q.p == 1.5


def test_multiple_instances():
    Q = State(p=0)
    W = State(p=0)

    Q.p = 1.5

    assert not(W.p == Q.p)
    assert Q.p == 1.5
    assert W.p == 0.0


def test_view(Q):
    np_arr = Q.view(np.ndarray)

    assert len(np_arr) == 3
    assert not np_arr.any()


def test_numpy_behaviour():
    X = State(x=1, y=0, z=0)
    Y = State(x=0, y=1, z=0)

    assert np.array_equal(np.cross(X, Y), np.array([0, 0, 1]))
    assert np.array_equal(X - Y, np.array([1, -1, 0]))
    assert X.dot(Y) == 0
