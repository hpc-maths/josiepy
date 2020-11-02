import numpy as np

from josie.boundary import CircleArc, Line


def test_circle_arc(tol):
    p0 = np.array([-1, 0])
    p1 = np.array([0, 1])
    p2 = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2])

    arc = CircleArc(p0, p1, p2)

    assert arc._r - 1 < tol


def test_line():
    p0 = np.array([-1, 0])
    p1 = np.array([0, 1])
    line = Line(p0, p1)

    assert line
