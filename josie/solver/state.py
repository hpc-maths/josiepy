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

from enum import IntEnum
from typing import Collection, Type

from josie.mesh import Mesh


class State(np.ndarray):
    """ :class:`State` is a subclass of :class:`numpy.ndarray`. It behaves like
    a normal :class:`numpy.ndarray` except it has additional init methods to
    ease the usage

    A :class:`State` can be initialized using a :class:`StateTemplate`, or
    directly providing key-value arguments:

    >>> Q = StateTemplate("rho", "rhoU", "rhoV")
    >>> state = np.array([0, 1, 2]).view(Q)
    >>> assert state[state.fields.rho] == 0
    >>> assert state[state.fields.rhoU] == 1
    >>> assert state[state.fields.rhoV] == 2

    A state can be manipulated as a normal :class:`numpy.ndarray`:

    >>> e1 = State([1, 0, 0])
    >>> e2 = State([0, 1, 0])
    >>> e3 = State([0, 0, 1])
    >>> assert np.array_equal(np.cross(e1, e2), e3)

    A :class:`State` can be multidimensional. The last dimension must be the
    number of states defined in the :class:`StateTemplate` call. In this case
    you can get all the values of the state for a specific variable:

    >>> state = np.random.random((10, 10, 3)).view(Q)
    >>> assert np.array_equal(state[..., state.fields.rho], state[..., 0])
    >>> assert np.array_equal(state[..., state.fields.rhoU],state[..., 1])
    """

    fields: Type[IntEnum]
    _FIELDS_ENUM_NAME = "FieldsEnum"

    def __new__(cls, *args):
        arr: State = np.asarray(list(args), dtype=float).view(cls)

        return arr

    def __array_finalize__(self, obj):
        # Normal __new__ construction
        if obj is None:
            return

        self._getitem = False

        # Slice handled by superclass
        if isinstance(obj, State) and obj._getitem:
            return

    @classmethod
    def list_to_enum(cls, fields: Collection[str]) -> IntEnum:
        """ Convert a list of textual fields to the class:`IntEnum` that needs
        to be stored in this class :attr:`fields` """

        return IntEnum(
            cls._FIELDS_ENUM_NAME, dict(zip(fields, range(len(fields))))
        )

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> State:
        """ Initialize an empty class:`State` object of the right dimensiosn
        for the given class:`Mesh` """

        nx = mesh.num_cells_x
        ny = mesh.num_cells_y
        state_size = len(cls.fields)

        # TODO: Fix for 3D. Probably adding a dimensionality attribute to
        # `Mesh`
        return np.empty((nx + 2, ny + 2, state_size)).view(cls)


def StateTemplate(*fields: str) -> Type[State]:
    r""" A factory for a :class:`State`.

    It allows you to create at will a :class:`State` class for which you can
    access its variables (e.g. the velocity :math:`\mathbf{U}`) using the
    attribute :attr:`fields`, that is an :class:`IntEnum` (and not only by
    index).

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
    >>> assert zero[Q.fields.rho] == 0
    """
    # Dynamically create a class of type "State" (actually a subclass)
    # with the right :attr:`fields`
    enum_fields: IntEnum = State.list_to_enum(fields)
    state_cls = type("DerivedState", (State,), {"fields": enum_fields})

    return state_cls
