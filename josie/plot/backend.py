# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" Backends used to display mesh and mesh results """
from __future__ import annotations

import abc

from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from josie.mesh import Mesh
    from josie.solver import Solver


class PlotBackend(metaclass=abc.ABCMeta):
    """An abstract interface representing a plot backend"""

    @abc.abstractmethod
    def plot(self, mesh: Mesh):
        """Allocate a drawing instance in order to draw a single image plot.
        The drawing state is stored in :attr:`plot_state`.

        Parameters
        ----------
        mesh
            The :class:`Mesh` to be displayed
        """

        raise NotImplementedError

    @abc.abstractmethod
    def append(self, solver: Solver, t):
        """Appends a new simulation time state

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
        """Updates the :attr:`plot_state` with the state (i.e. the field data
        stored in the mesh, e.g. cell data) in the Solver.

        By default each call to :func:`update` overrides the plot state (in
        order to save up memory).

        Parameters
        ----------
        solver
            An instance of :class:`Solver` that stores the current state of the
            simulation
        """

        raise NotImplementedError

    @abc.abstractmethod
    def show(self, fields: Union[List, str]):
        """Show on screen a list of fields.

        Parameters
        ----------
        fields
            The list of fields to show. If `None`, then only the mesh is
            displayed
        """

        raise NotImplementedError

    @abc.abstractmethod
    def show_grid(self):
        """Show the grid on screen"""
        raise NotImplementedError

    @abc.abstractmethod
    def show_all(self):
        """Show on screen all the fields"""

        raise NotImplementedError
