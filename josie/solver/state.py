# josiepy
# Copyright © 2019 Ruben Di Battista
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Ruben Di Battista ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Ruben Di Battista BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Ruben Di Battista.

import numpy as np

from collections import OrderedDict


class _StateDescriptor:
    """ This is a custom descriptor to be used within the State class. It
    provides the possibilidy of accessing the ith-element of the numpy.ndarray
    by name.

    Parameters
    ---------
    i
        The index of the element in the numpy array to be accessed with

    """
    def __init__(self, i):
        self.i = i

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj[self.i]

    def __set__(self, obj, value):
        obj[self.i] = value


class StateTemplate:
    """ A factory for a :class:`State`. It encapsulates the number of variables
    composing the :class:`State` and it allows to concretize a state as if it
    was an object instantiation

    Parameters
    ----------
    fields
        A list of (scalar) fields composing the state

    Attributes
    ----------
    fields
        The list of (scalar) fields composing the state

    A scalar :class:`State` as for the advection equation
    >>> Q = StateTemplate("u")

    Than you can concretize the state with a value
    >>> zero = Q(0)

    You can also create higher dimensional states, for examples the state
    of the 2D Euler compressible equations
    >>> Q = StateTemplate("rho", "rhoU", "rhoV", "E")
    >>> zero = Q(0, 0, 0, 0)
    """

    def __init__(self, *fields):
        self.fields = fields

    def __call__(self, *values):
        if not(len(values) == len(self.fields)):
            raise ValueError(f"This state has {len(self.fields)} fields. "
                             "You need to provide the same amount of values")

        d = OrderedDict(zip(self.fields, values))
        return State(**d)


class State(np.ndarray):
    """ :class:`State` is a subclass of :class:`numpy.ndarray`. It behaves like
    a normal :class:`numpy.ndarray` except it can be initialized a bit more
    expressively. In particular each element of the array can be accessed by
    name (in the order variables are provided)

    Parameters
    ----------
    variables
        keyword arguments whose keys are the variables names and the values
        their values

    Attributes
    ----------
    variables
        Each key of the kwargs provided as input are now attributes
        pointing to the right element in the array


    A :class:`State` can be initialized using a :class:`StateTemplate`, or
    directly providing key-value arguments:

    >>> Q = State(rho=0, rhoU=1, rhoV=2)
    >>> assert Q.rho == 0
    >>> assert Q.rhoU == 1
    >>> assert Q.rhoV == 2
    >>> assert Q[0] == Q.rho
    >>> assert Q[1] == Q.rhoU
    >>> assert Q[2] == Q.rhoV

    A state can be manipulated as a normal :class:`numpy.ndarray`:

    >>> e1 = State(i=1, j=0, k=0)
    >>> e2 = State(i=0, j=1, k=0)
    >>> e3 = State(i=0, j=0, k=1)
    >>> assert np.array_equal(np.cross(e1, e2), e3)
    """

    def __new__(cls, **variables):
        for i, field in enumerate(variables.keys()):
            setattr(cls, field, _StateDescriptor(i))

        arr = np.asarray(list(variables.values()), dtype=float).view(cls)

        return arr

    def __array_finalize(self, obj):
        if obj is None:
            return
