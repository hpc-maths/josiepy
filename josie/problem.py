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

import abc
import numpy as np

from typing import Union

from josie.mesh.cellset import CellSet, MeshCellSet

from josie.state import State


class Problem(metaclass=abc.ABCMeta):
    r"""A class representing a physical problem to be solved (governed by
    a PDE).

    A general problem can be written in a compact way:

    .. math::

        \pdeFull

    A concrete instance of this class potentially provides the terms

    .. math::

        \pdeTermList
    """

    def __init__(self, **kwargs):
        super().__init__()

    def F(self, cells: Union[CellSet, MeshCellSet]) -> np.ndarray:
        r"""The convective flux operator :math:`\pdeConvective`

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` that contains the cell data

        Returns
        -------
        F
            An array of dimension :math:`Nx \times Ny \times N_\text{eqs}
            \times N_\text{dim}` containing the computed convective flux for
            the given problem
        """
        pass

    def B(self, cells: Union[CellSet, MeshCellSet]) -> np.ndarray:
        r"""This returns the tensor that pre-multiplies the non-conservative
        term of the problem.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull


        This method needs to return :math:`\pdeNonConservativeMultiplier`

        Parameters
        ----------
        state_array
            The :class:`State` array containing the values of all the fields

        Returns
        -------
        B
            An array of dimension :math:`Nx \times Ny \times N_\text{eqs}
            \times N_\text{state} \times N_\text{dim}` containing the computed
            pre-multiplier tensor
        """
        pass

    def K(self, cells: Union[CellSet, MeshCellSet]) -> np.ndarray:
        r"""This returns the tensor that pre-multiplies the gradient in the
        diffusive term of the problem.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull


        This method needs to return :math:`\pdeDiffusiveMultiplier`

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` that contains the cell data

        Returns
        -------
        D
            An array of dimension :math:`Nx \times Ny \times N_\text{eqs}
            \times N_\text{state} \times N_\text{dim} \times N_\text{dim}`
            containing the computed pre-multiplier tensor
        """
        pass

    def s(self, cells: Union[CellSet, MeshCellSet], t: float) -> State:
        r"""This returns the values of the source terms

        A general problem can be written in a compact way:

        .. math::

            \pdeFull


        This method needs to return :math:`\pdeSource`

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` that contains the cell data

        t
            Time instant

        Returns
        -------
        s
            An array of dimension :math:`Nx \times Ny \times N_\text{eqs}`
            containing the computed source terms
        """
        pass
