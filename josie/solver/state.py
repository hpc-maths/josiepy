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
        num_fields = len(variables)

        for i, field in enumerate(variables.keys()):
            setattr(cls, field, _StateDescriptor(i))

        arr = np.asarray(list(variables.values()), dtype=float).view(cls)

        return arr

    def __array_finalize(self, obj):
        if obj is None:
            return
