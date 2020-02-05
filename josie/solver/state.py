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

from collections import Sequence
from typing import Type, Tuple, Union


class _StateDescriptor:
    """ This is a custom descriptor to be used within the State class. It
    provides the possibility of accessing the ith-element of the numpy.ndarray
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

    _variables: Tuple[str]

    def __new__(cls, *args, **kwargs):
        if args and kwargs:
            raise ValueError(
                "You can initialize a state using positional arguments only "
                "or keyword arguments only. Not both"
            )

        if kwargs:
            cls._variables = tuple(kwargs.keys())
            values = kwargs.values()
        else:
            if not (len(args) == len(cls._variables)):
                raise ValueError(
                    "The number of provided input arguments must be the same "
                    "as the number of the variables of the state"
                )

            values = args

        cls._set_descriptors(cls)

        arr = np.asarray(list(values), dtype=float).view(cls)

        return arr

    def __getitem__(self, item):
        self._slice_index = item

        return super().__getitem__(item)

    def __array_finalize__(self, obj):
        # Normal __new__ construction
        if obj is None:
            return
        else:  # Here we are a view or a new-from-template
            # If we are a new-from-template, need also to slice the variables
            # to the one selected by the slice
            if isinstance(self, State) and (isinstance(obj, State)):
                cls = self.__class__
                if isinstance(obj._slice_index, Sequence):
                    # cls._variables is a tuple and can be indexed only
                    # by integer or slice. With a sequence we manually select
                    # the values specified by the list
                    new_variables = []

                    # First we need to get the actual variables that need are
                    # requested by the "slice"
                    for i in obj._slice_index:
                        new_variables.append(cls._variables[i])

                    new_variables = tuple(new_variables)
                else:
                    new_variables = self._variables[obj._slice_index]

                self._sync_descriptors(new_variables)
            self._set_descriptors(self)

    @staticmethod
    def _set_descriptors(obj: Union[Type["State"], "State"]):
        """ This method sets the descriptor to each variable of the state
        provided by :class:_StateDescriptor for the class. Can be called
        on the class or on a instance of :class:`State`
        """

        if isinstance(obj, State):
            cls = obj.__class__
        else:
            cls = obj

        for i, field in enumerate(obj._variables):
            setattr(cls, field, _StateDescriptor(i))

    def _sync_descriptors(self, new_variables: Tuple[str]):
        """ This method syncs the descriptors removing unused variables """
        # We check on the full set of variables of the State
        # which ones are requested by the "slice"
        cls = self.__class__
        for old_var in self._variables:
            if not (old_var in new_variables):  # Requested
                # We remove the associated attribute
                delattr(cls, old_var)

        self._variables = new_variables


class StateTemplate:
    r""" A factory for a :class:`State`.

    It allows you to create at will a :class:`State` class for which you can
    access its variables (e.g. the velocity :math:`\mathbf{U}`) by attributes
    (and not only by index).

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
    >>> assert Q.rho == 0
    """

    def __new__(cls, *fields: str) -> State:
        state_cls = State
        state_cls._variables = fields

        return state_cls
