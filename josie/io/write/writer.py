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
import copy
import os

from typing import List

from meshio.xdmf import TimeSeriesWriter

from josie.solver.solver import Solver
from josie.data import StateElement

from .strategy import NoopStrategy, Strategy


class Writer(abc.ABC):
    """ A context manager to apply a writing strategy for a simulation

    Child classes implement writing every :math:`n_t` time-steps or every
    :math:`n` iterations and so on....

    Attributes
    ----------
    strategy
        An instance of :class:`~.Strategy` that implements a serializing
        strategy
    solver
        An instance of the solver to manage the execution of
    final_time
        The final time in seconds at which the simulation must end
    CFL
        The value of the CFL number to limit the time stepping of a specific
        scheme
    """

    def __init__(
        self,
        strategy: Strategy,
        solver: Solver,
        final_time: float,
        CFL: float,
    ):
        self._t = 0

        self.strategy = strategy
        self.solver = solver
        self.final_time = final_time
        self.CFL = CFL

    @abc.abstractmethod
    def write(self):
        """ This methods serializes the solver state to disk or else
        """

        pass

    def solve(self):
        """ This method updates the internal state of the Writer and saves
        solver state to the file if needed
        """

        solver = self.solver

        while self._t < self.final_time:

            dt = solver.scheme.CFL(
                solver.values,
                solver.mesh.volumes,
                solver.mesh.surfaces,
                self.CFL,
            )

            dt = self.strategy.check_write(self._t, dt, solver)

            if self.strategy.should_write:
                self.write()
                if self.strategy.animate:
                    solver.animate(self._t)

            solver.step(dt)

            self._t += dt


class NoopWriter(Writer):
    """ A :class:`Writer` that does not write anything """

    def __init__(self, solver: Solver, final_time: float, CFL: float):
        super().__init__(NoopStrategy(), solver, final_time, CFL)

    def write(self):
        pass


class FileWriter(Writer):
    """ This abstract class provides an interface to write to a file

    Attributes
    ----------
    filename
        A :class:`pathlib.Path` to the file to which the data needs to be
        serialized to
    strategy
        An instance of :class:`~.Strategy` that implements a serializing
        strategy
    solver
        An instance of the solver to manage the execution of
    final_time
        The final time in seconds at which the simulation must end
    CFL
        The value of the CFL number to limit the time stepping of a specific
        scheme
    """

    def __init__(
        self,
        filename: os.PathLike,
        strategy: Strategy,
        solver: Solver,
        final_time: float,
        CFL: float,
    ):
        super().__init__(strategy, solver, final_time, CFL)

        self.filename = filename


class MemoryWriter(Writer):
    """ This class provides serialization of simulation data into a
    list of :class:`~.StateElement`

    Attributes
    ----------
    data
        The simulation data. A list of :class:`~.StateElement`
    strategy
        An instance of :class:`~.Strategy` that implements a serializing
        strategy
    solver
        An instance of the solver to manage the execution of
    final_time
        The final time in seconds at which the simulation must end
    CFL
        The value of the CFL number to limit the time stepping of a specific
        scheme
    """

    def __init__(
        self, strategy: Strategy, solver: Solver, final_time: float, CFL: float
    ):
        super().__init__(strategy, solver, final_time, CFL)

        self.store: List[StateElement] = []

    def write(self):
        data = {}

        # TODO: If mesh becomes "dynamic", this probably needs to be deepcopied
        data["mesh"] = self.solver.mesh

        for field in self.solver.Q.fields:
            data[field.name] = copy.deepcopy(
                self.solver.values[..., field].ravel()
            )

        self.store.append(StateElement(time=self._t, data=data))


class XDMFWriter(FileWriter):
    """ A class allowing to serialize simulation data to an XDMF time-series

    Attributes
    ----------
    filename
        A :class:`pathlib.Path` to the file to which the data needs to be
        serialized to
    strategy
        An instance of :class:`~.Strategy` that implements a serializing
        strategy
    solver
        An instance of the solver to manage the execution of
    final_time
        The final time in seconds at which the simulation must end
    CFL
        The value of the CFL number to limit the time stepping of a specific
        scheme
    """

    def write(self):
        io_mesh = self.solver.mesh.export()
        self._writer.write_points_cells(io_mesh.points, io_mesh.cells)

        cell_data = {}

        cell_type_str = self.solver.mesh.cell_type._meshio_cell_type

        for field in self.solver.Q.fields:
            cell_data[field.name] = {
                cell_type_str: self.solver.values[..., field].ravel()
            }

        self._writer.write_data(self._t, cell_data=cell_data)

    def solve(self):
        with TimeSeriesWriter(self.filename) as self._writer:
            super().solve()
