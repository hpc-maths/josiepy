# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import numpy as np
import pytest

from josie.state import State, StateTemplate


@pytest.fixture
def QTemplate():
    yield StateTemplate("rho", "rhoU", "rhoV", "p")


@pytest.fixture()
def Q(QTemplate):
    yield QTemplate(0, 0, 0, 0)


@pytest.fixture
def multiQ(QTemplate):
    yield np.random.random((10, 10, 4)).view(QTemplate)


def test_list_to_enum():
    Fields = State.list_to_enum(["rho", "rhoU"])

    assert Fields.rho == 0
    assert Fields.rhoU == 1


def test_pickling(Q):
    restored_state = pickle.loads(pickle.dumps(Q))

    assert np.array_equal(restored_state, Q)
    assert [f for f in restored_state.fields] == [f for f in Q.fields]


def test_get_attributes(Q):
    assert Q[Q.fields.rhoU] == 0
    assert Q[Q.fields.rhoV] == 0
    assert Q[Q.fields.p] == 0


def test_set_attributes(Q):
    Q[Q.fields.p] = 1.5

    assert Q[Q.fields.p] == 1.5


def test_multiple_instances(QTemplate):
    Q = QTemplate(0, 0, 0, 0)
    W = QTemplate(0, 0, 0, 0)

    Q[Q.fields.p] = 1.5

    assert not (W[W.fields.p] == Q[Q.fields.p])
    assert Q[Q.fields.p] == 1.5
    assert W[W.fields.p] == 0.0


def test_view(Q):
    np_arr = Q.view(np.ndarray)

    assert len(np_arr) == 4
    assert np.array_equal(np_arr, Q)


def test_reverse_view(QTemplate):
    arr = np.array([0, 1, 2, 3])
    Q = arr.view(QTemplate)

    assert Q[Q.fields.rho] == 0
    assert Q[Q.fields.rhoU] == 1
    assert Q[Q.fields.rhoV] == 2
    assert Q[Q.fields.p] == 3


def test_numpy_behaviour():
    Q = StateTemplate("x", "y", "z")
    X = Q(1, 0, 0)
    Y = Q(0, 1, 0)

    assert np.array_equal(np.cross(X, Y), np.array([0, 0, 1]))
    assert np.array_equal(X - Y, np.array([1, -1, 0]))
    assert X.dot(Y) == 0

    assert X[X.fields.x] == 1
    assert X[X.fields.y] == 0
    assert X[X.fields.z] == 0

    assert Y[Y.fields.x] == 0
    assert Y[Y.fields.y] == 1
    assert Y[Y.fields.z] == 0


def test_broadcasting(Q):
    a = np.zeros((5, len(Q.fields)))

    Q - a


def test_multidimensional_numpy_behaviour():
    Q = StateTemplate("x", "y", "z")

    X = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]).view(Q)

    Y = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]).view(Q)

    Z = np.cross(X, Y)

    assert np.all(Z == np.array([0, 0, 1]))
    assert np.all(X - Y == np.array([1, -1, 0]))
    assert np.all(np.einsum("ij,ij->i", X, Y) == 0)
