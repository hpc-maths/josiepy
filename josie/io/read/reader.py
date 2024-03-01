# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import abc

import logging

from typing import Type

from meshio.xdmf import TimeSeriesReader

from josie.state import State

logger = logging.getLogger(__name__)


class FileReader(abc.ABC):
    """A context manager to read the results of a simulation

    Attributes
    ----------
    filename
        Name of the file where the data is stored
    """

    def __init__(
        self,
        filename: str,
        Q: Type[State],
    ):
        self.filename = filename
        self.Q = Q

    @abc.abstractmethod
    def read(self, num_step: int, values: State):
        """This methods reads the state from a file"""

        pass

    @abc.abstractmethod
    def read_dim(self) -> int:
        pass

    @abc.abstractmethod
    def read_time(self, num_step: int) -> float:
        pass


class XDMFReader(FileReader):
    """A class allowing to read simulation data from an XDMF time-series

    Attributes
    ----------
    filename
        A :class:`pathlib.Path` to the file to which the data needs to be
        read
    """

    def read_dim(self) -> int:
        with TimeSeriesReader(self.filename) as reader:
            _, cells = reader.read_points_cells()
            return cells[0].data.shape[0]

    def read_time(self, num_step: int) -> float:
        with TimeSeriesReader(self.filename) as reader:
            _, _ = reader.read_points_cells()
            t, _, _ = reader.read_data(num_step)
            return t

    def read(self, num_step: int, values: State):
        with TimeSeriesReader(self.filename) as reader:
            _, _ = reader.read_points_cells()
            _, _, cell_data = reader.read_data(num_step)

            for field in self.Q.fields:
                values[..., 0, field] = cell_data[field.name][0].reshape(
                    values[..., 0, field].shape
                )
