import abc

import matplotlib.pyplot as plt
import numpy as np


def map01to(x, a, b):
    """ Maps x in [0, 1] to [a, b] """

    return (b - a)*x + a


class BoundaryCurve(metaclass=abc.ABCMeta):
    r""" A class representing a BoundaryCurve. A BoundaryCurve is parametrized
    with a single parameter. It implements a `__call__` method to that returns
    the :math:`(x,y)` values of the curve for a given :math:`\xi`
    parameter value.
    """

    @abc.abstractmethod
    def __call__(self, xi):
        """ The effective parametrization of the BoundaryCurve. Assume `xi` to
        range into [0, 1]

        Args:
            xi: The parameter the curve is parametrized with.
                Ranges from 0 to 1
        """

        raise NotImplementedError

    def plot(self, resolution=50):
        """ This method actually plots the BoundaryCurve

        This method currently plots stuff using matplotlib. It generates the
        list of points to plot with a default `resolution`.

        Args:
            resolution: The number of points to plot the curve with
                [default: 50].
        """

        xi = np.linspace(0, 1, resolution)

        X, Y = self(xi)

        plt.plot(X, Y, 'o-')


class CircleArc(BoundaryCurve):
    """A class representing a circular arc from two given points and the
    center of the arc.

    Parameters:
        p1: Starting point of the arc
        p2: Ending point of the arc
        center: Center of the arc
        reverse: Plot the reverse side of the arc [default: False]
    """

    def __init__(self, p1, p2, center, reverse=False):
        self._p1 = p1
        self._p2 = p2
        self._c = center
        self._r1 = self._p1 - self._c
        self._r2 = self._p2 - self._c

        try:
            assert(np.linalg.norm(self._r1) - np.linalg.norm(self._r2) < 1E-12)
            self._r = np.linalg.norm(self._r1)

        except AssertionError:
            raise ValueError("The two points are not at the same distance "
                             "from the center point")

        # Starting angles
        cos1 = self._r1[0]/self._r
        cos2 = self._r2[0]/self._r

        sin1 = self._r1[1]/self._r
        sin2 = self._r2[1]/self._r

        th1 = np.arctan2(sin1, cos1)
        th2 = np.arctan2(sin2, cos2)

        if not(reverse):
            th1 = (th1 + 2*np.pi) % (2*np.pi)
            th2 = (th2 + 2*np.pi) % (2*np.pi)

        # Sort angles
        angles = np.sort([th1, th2])

        self._th1 = angles[0]
        self._th2 = angles[1]

    def __call__(self, xi):

        # Remap to the correct angle range
        xi = map01to(xi, self._th1, self._th2)

        return (self._c[0] + self._r*np.cos(xi),
                self._c[1] + self._r*np.sin(xi))


if __name__ == '__main__':
    p0 = np.array([0, 1])
    p1 = np.array([-1, 0])
    pc = np.array([0, 0])
    pc2 = pc + 0.4*np.array([-1, 1])
    arc = CircleArc(p0, p1, pc)
    arc2 = CircleArc(p0, p1, pc2, neg=True)
    plt.plot(p0[0], p0[1], 'o')
    plt.plot(p1[0], p1[1], 'o')
    plt.plot(pc[0], pc[1], 'o')
    plt.plot(pc2[0], pc2[1], 'o')
    arc.plot()
    arc2.plot()

    plt.axis('equal')
    plt.show()
