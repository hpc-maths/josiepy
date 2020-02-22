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
from __future__ import annotations

import numpy as np

from collections.abc import Sequence
from typing import Type, Union


class _StateDescriptor:
    """ This is a custom descriptor to be used within :class:`State`. It
    provides the possibility of accessing the ith-element of the numpy.ndarray
    by name.

    Parameters
    ---------
    i
        The index of the element in the numpy array to be accessed with

    """

    def __init__(self, i):
        self.i = i
        self._deleted = False

    def __get__(self, obj: State, objtype=None):
        if obj is None:
            return self
        if self._deleted:
            raise AttributeError("Attribute was deleted")

        ret = obj[..., self.i]

        if ret.ndim == 0:
            ret = ret.item(0)

        return ret

    def __set__(self, obj, value):
        obj[..., self.i] = value

    def __delete__(self, obj):
        self._deleted = True


class State(np.ndarray):
    """ :class:`State` is a subclass of :class:`numpy.ndarray`. It behaves like
    a normal :class:`numpy.ndarray` except it can be initialized a bit more
    expressively. In particular each element of the array can be accessed by
    name

    Attributes
    ----------
    fields
        Each element of the tuple is the name of the corresponding element in
        the array


    A :class:`State` can be initialized using a :class:`StateTemplate`, or
    directly providing key-value arguments:

    >>> Q = StateTemplate("rho", "rhoU", "rhoV")
    >>> state = np.array([0, 1, 2]).view(Q)
    >>> assert state.rho == 0
    >>> assert state.rhoU == 1
    >>> assert state.rhoV == 2
    >>> assert state[0] == state.rho
    >>> assert state[1] == state.rhoU
    >>> assert state[2] == state.rhoV

    A state can be manipulated as a normal :class:`numpy.ndarray`:

    >>> e1 = State(i=1, j=0, k=0)
    >>> e2 = State(i=0, j=1, k=0)
    >>> e3 = State(i=0, j=0, k=1)
    >>> assert np.array_equal(np.cross(e1, e2), e3)

    A :class:`State` can be multidimensional. The last dimension must be the
    number of states defined in the :class:`StateTemplate` call. In this case
    you can get all the values of the state for a specific variable:

    >>> state = np.random.random((10, 10, 3)).view(Q)
    >>> assert state.rho == state[..., 0]
    >>> assert state.rhoU == state[..., 1]
    """

    fields: np.ndarray[str]

    def __new__(cls, *args, **kwargs):
        if args and kwargs:
            raise ValueError(
                "You can initialize a state using positional arguments only "
                "or keyword arguments only. Not both"
            )

        if kwargs:
            cls.fields = np.array(list(kwargs.keys()), ndmin=1)
            values = kwargs.values()
        else:
            if not (len(args) == len(cls.fields)):
                raise ValueError(
                    "The number of provided input arguments must be the same "
                    "as the number of the variables of the state"
                )

            values = args

        cls._set_descriptors(cls)

        arr = np.asarray(list(values), dtype=float).view(cls)

        return arr

    def __getitem__(self, item):
        self._getitem = True

        try:
            ret: np.ndarray = super().__getitem__(item)
        finally:
            self.__getitem__ = False

        if not isinstance(ret, np.ndarray):
            return ret

        self._slice_index = item

        if isinstance(self._slice_index, Sequence):
            # If it's an Ellipsis we're within the __get__ call of the
            # descriptor. So we noop.
            if self._slice_index[0] is Ellipsis:
                return ret

            # If the first element is a slice, we're doing a call of the
            # type array[:, 0]. So we use the second element as index for
            # the fields
            if isinstance(self._slice_index[0], slice):
                self._slice_index = self._slice_index[0]

        newfields = np.atleast_1d(self.fields[self._slice_index])

        self._sync_descriptors(ret, newfields)
        self._set_descriptors(ret)

        return ret

    def __array_finalize__(self, obj):
        # Normal __new__ construction
        if obj is None:
            return

        self._getitem = False

        # Slice handled by superclass
        if isinstance(obj, State) and obj._getitem:
            return
        else:
            # View
            dims = obj.shape
            if not (dims[-1] == len(self.fields)):
                raise ValueError(
                    "View has different number of elements than fields "
                )

        self._set_descriptors(self)

    @staticmethod
    def _set_descriptors(obj: Union[Type[State], State]):
        """ This method sets the descriptor to each variable of the state
        provided by :class:_StateDescriptor for the class. Can be called
        on the class or on a instance of :class:`State`
        """

        if isinstance(obj, State):
            cls = obj.__class__
        else:
            cls = obj

        for i, field in enumerate(obj.fields):
            setattr(cls, field, _StateDescriptor(i))

    @staticmethod
    def _sync_descriptors(obj, newfields: np.ndarray[str]):
        """ This method syncs the descriptors removing unused fields"""
        # We check on the full set of fields of the State
        # which ones are requested by the "slice"
        for old_field in obj.fields:
            if not (old_field in newfields):  # Requested
                # We remove the associated attribute
                delattr(obj, old_field)

        obj.fields = newfields

    @classmethod
    def zeros(cls):
        return np.zeros(len(cls.fields))


def StateTemplate(*fields: str) -> Type[State]:
    r""" A factory for a :class:`State`.

    It allows you to create at will a :class:`State` class for which you can
    access its variables (e.g. the velocity :math:`\mathbf{U}`) by attributes
    (and not only by index).

    Parameters
    ----------
    fields
        A list of (scalar) fields composing the state

    A scalar :class:`State` as for the advection equation
    >>> Q = StateTemplate("u")

    Than you can concretize the state with a value
    >>> zero = Q(0)

    You can also create higher dimensional states, for examples the state
    of the 2D Euler compressible equations
    >>> Q = StateTemplate("rho", "rhoU", "rhoV", "E")
    >>> zero = Q(0, 0, 0, 0)
    >>> assert zero.rho == 0
    """
    # Dynamically create a class of type "State" (actually a subclass)
    # with the right :attr:`fields`
    fields = np.array(fields, ndmin=1)
    state_cls = type("DerivedState", (State,), {"fields": fields})

    return state_cls
