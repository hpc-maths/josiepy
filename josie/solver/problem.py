import abc

from .state import State


class Problem(metaclass=abc.ABCMeta):

    @property
    @abc.abstractclassmethod
    def Q(cls):
        raise NotImplementedError

    @abc.abstractclassmethod
    def flux(cls, Q: State) -> State:
        raise NotImplementedError
