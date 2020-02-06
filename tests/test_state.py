import numpy as np
import pytest

from josie.solver.state import State, StateTemplate


def state_from_template():
    QTemplate = StateTemplate("rho", "rhoU", "rhoV", "p")
    return QTemplate(0, 0, 0, 0)


def state_directly():
    return State(rho=0, rhoU=0, rhoV=0, p=0)


@pytest.fixture(params=[state_from_template, state_directly])
def Q(request):
    yield request.param()


def test_wrong_number_of_fields():
    Q = StateTemplate("rhoU", "rhoV", "p")
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

    assert not (W.p == Q.p)
    assert Q.p == 1.5
    assert W.p == 0.0


def test_list_getitem(Q):
    Q.rhoV = 12
    Q_slice = Q[[0, 2]]

    with pytest.raises(AttributeError):
        Q_slice.rhoU
        Q_slice.p

    assert Q_slice.rho == 0
    assert Q_slice.rhoV == 12


def test_slice_getitem(Q):
    Q.rhoV = 12
    Q_slice = Q[:3]

    with pytest.raises(AttributeError):
        Q_slice.p

    assert Q_slice.rhoV == 12
    assert Q_slice.rhoV == Q_slice[-1]


def test_view(Q):
    np_arr = Q.view(np.ndarray)

    assert len(np_arr) == 4
    assert np.array_equal(np_arr, Q)


def test_reverse_view():
    QTemplate = StateTemplate("rho", "rhoU", "rhoV", "p")
    arr = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        Q = arr.view(QTemplate)

    arr = np.array([0, 1, 2, 3])
    Q = arr.view(QTemplate)

    Q.rho
    Q.rhoU
    Q.rhoV
    Q.p


def test_numpy_behaviour():
    X = State(x=1, y=0, z=0)
    Y = State(x=0, y=1, z=0)

    assert np.array_equal(np.cross(X, Y), np.array([0, 0, 1]))
    assert np.array_equal(X - Y, np.array([1, -1, 0]))
    assert X.dot(Y) == 0

    assert X.x == 1
    assert X.y == 0
    assert X.z == 0

    assert Y.x == 0
    assert Y.y == 1
    assert Y.z == 0


def test_broadcasting(Q):
    a = np.zeros((5, len(Q.fields)))

    Q - a
