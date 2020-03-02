import numpy as np
import pytest

from josie.solver.twophase.state import Q


@pytest.fixture
def state():
    yield np.array(range(len(Q.fields))).view(Q)


def test_phase(state):
    __import__("ipdb").set_trace()
