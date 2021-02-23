# josiepy
# Copyright © 2020 Ruben Di Battista
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

import abc

from josie.solver import Solver


class Strategy(abc.ABC):
    """The write strategy to use when serializing a simulation.

    Attributes
    ----------
    should_write
        If True, solver must be serialized.
    animate
        If True, enables capturing an animation with this strategy
    serialize
        If True, enables serializing the solver state with this strategy
    """

    def __init__(self, animate: bool = False, serialize: bool = True):
        self.should_write = False
        self.animate = animate
        self.serialize = serialize

    @abc.abstractmethod
    def check_write(self, t: float, dt: float, solver: Solver) -> float:
        """Updates the state and modifies the :attr:`should_write` if needed.

        Returns
        -------
        dt
            A possibly modified dt
        """

        return dt


class NoopStrategy(Strategy):
    """A :class:`Strategy` that never serializes anything"""

    def __init__(self):
        super().__init__(False, False)

    def check_write(self, t: float, dt: float, solver: Solver):
        return super().check_write(t, dt, solver)


class TimeStrategy(Strategy):
    """A :class:`Strategy` that serializes every `dt_save` seconds of
    simulated time

    Attributes
    ---------
    dt_save
        The time interval after which the solver state must be serialized
    t_save
        Last time solver was serialized
    """

    def __init__(
        self,
        dt_save: float,
        animate: bool = False,
        serialize: bool = True,
    ):
        super().__init__(animate, serialize)

        self.dt_save = dt_save

        # First slot for saving is 0 + dt_save
        self.t_save = dt_save

    def check_write(self, t: float, dt: float, solver: Solver) -> float:
        self.should_write = False

        if t == 0:
            # Always write initial instant
            self.should_write = True
            return dt

        if t + dt > self.t_save:
            self.should_write = True

            # We constrain the dt to respect the t_save
            dt = self.t_save - t

            # Next time we serialize
            self.t_save = self.t_save + self.dt_save

        return dt


class IterationStrategy(Strategy):
    """A :class:`Strategy` that serializes every `n` iterations

    Attributes
    ---------
    n
        The number of iterations after which we need to serialize
    """

    def __init__(self, n: int, animate: bool = False, serialize: bool = True):
        super().__init__(animate, serialize)

        self.n = n
        self._it: int = 0

    def check_write(self, t: float, dt: float, solver: Solver) -> float:
        dt = super().check_write(t, dt, solver)

        self._it += 1

        if self._it % self.n:
            self.should_write = True

        return dt
