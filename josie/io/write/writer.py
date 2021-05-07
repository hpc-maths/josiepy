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
import os

import logging

from typing import List

from meshio.xdmf import TimeSeriesWriter

from josie.solver import Solver

from .strategy import NoopStrategy, Strategy

logger = logging.getLogger(__name__)


class Writer(abc.ABC):
    """A context manager to apply a writing strategy for a simulation

    Child classes implement writing on files or in memory.

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
        self.strategy = strategy
        self.solver = solver
        self.final_time = final_time
        self.CFL = CFL

    @abc.abstractmethod
    def write(self):
        """This methods serializes the solver state to disk or else"""

        pass

    def solve(self):
        """This method updates the time instant in the :class:`Solver`, the
        internal state of the Writer and saves current solver state to the file
        if needed
        """
        logger.info("Solving...")

        solver = self.solver

        while self.solver.t < self.final_time:
            logger.info(f"Current time: {self.solver.t}")

            dt = solver.scheme.CFL(
                solver.mesh.cells,
                self.CFL,
            )

            dt = self.strategy.check_write(self.solver.t, dt, solver)

            if self.strategy.should_write:
                self.write()
                if self.strategy.animate:
                    # TODO: Factor out in separate object hierarchy the
                    # `animate` method of :class:`Solver`
                    solver.animate(self.solver.t)

            solver.step(dt)


class NoopWriter(Writer):
    """ A :class:`Writer` that does not write anything """

    def __init__(self, solver: Solver, final_time: float, CFL: float):
        super().__init__(NoopStrategy(), solver, final_time, CFL)

    def write(self):
        pass


class FileWriter(Writer):
    """This abstract class provides an interface to write to a file

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
    """This class provides serialization of simulation data into :class:`State`
    with an additional field storing time

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

        self.data: List[Solver] = []

    def write(self):
        self.data.append(self.solver.copy())


class XDMFWriter(FileWriter):
    """A class allowing to serialize simulation data to an XDMF time-series

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
                cell_type_str: self.solver.mesh.cells.values[
                    ..., field
                ].ravel()
            }

        self._writer.write_data(self.solver.t, cell_data=cell_data)

    def solve(self):
        with TimeSeriesWriter(self.filename) as self._writer:
            super().solve()
