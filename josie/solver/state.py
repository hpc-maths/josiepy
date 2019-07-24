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
    def __init__(self, i):
        self.i = i

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj[self.i]

    def __set__(self, obj, value):
        obj[self.i] = value


class StateTemplate:
    def __init__(self, *fields):
        self.fields = fields

    def __call__(self, *values):
        if not(len(values) == len(self.fields)):
            raise ValueError(f"This state has {len(self.fields)} fields. "
                             "You need to provide the same amount of values")

        d = OrderedDict(zip(self.fields, values))
        return State(**d)


class State(np.ndarray):
    """ The State class is basically an alias of a numpy recarray"""

    def __new__(cls, **variables):
        for i, field in enumerate(variables.keys()):
            setattr(cls, field, _StateDescriptor(i))

        arr = np.asarray(list(variables.values()), dtype=float).view(cls)

        return arr

    def __array_finalize(self, obj):
        if obj is None:
            return
