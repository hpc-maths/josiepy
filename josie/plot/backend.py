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
""" Backends used to display mesh and mesh results """
from __future__ import annotations

import abc

from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from josie.mesh import Mesh
    from josie.solver import Solver


class PlotBackend(metaclass=abc.ABCMeta):
    """ An abstract interface representing a plot backend
    """

    @abc.abstractmethod
    def plot(self, mesh: Mesh):
        """ Allocate a drawing instance in order to draw a single image plot.
        The drawing state is stored in :attr:`plot_state`.

        Parameters
        ----------
        mesh
            The :class:`Mesh` to be displayed
        """

        raise NotImplementedError

    @abc.abstractmethod
    def append(self, solver: Solver, t):
        """ Appends a new simulation time state

        Parameters
        ---------
        solver
            An instance of :class:`Solver` that stores the state of the
            simulation associated to the time instant `t`

        t
            The time instant of the simulation state
        """

        raise NotImplementedError

    @abc.abstractmethod
    def update(self, solver: Solver):
        """ Updates the :attr:`plot_state` with the state (i.e. the field data
        stored in the mesh, e.g. cell data) in the Solver.

        By default each call to :func:`update` overrides the plot state (in
        order to save up memory).

        Parameters
        ----------
        solver
            An instance of :class:`Solver` that stores the current state of the
            simulation
        t
            The time instant for which to store the state of the simulation.
            Useful if you want to produce an anumation instead of a static
            image.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def show(self, fields: Union[List, str]):
        """ Show on screen a list of fields.

        Parameters
        ----------
        fields
            The list of fields to show. If `None`, then only the mesh is
            displayed
        """

        raise NotImplementedError

    @abc.abstractmethod
    def show_grid(self):
        """ Show the grid on screen """
        raise NotImplementedError

    @abc.abstractmethod
    def show_all(self):
        """ Show on screen all the fields """

        raise NotImplementedError
