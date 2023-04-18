# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import numpy as np

from typing import Union

from josie.mesh.cellset import CellSet, MeshCellSet


# class Problem(metaclass=abc.ABCMeta):
#     r"""A class representing a physical problem to be solved (governed by
#     a PDE).

#     A general problem can be written in a compact way:

#     .. math::

#         \pdeFull

#     A concrete instance of this class potentially provides the terms

#     .. math::

#         \pdeTermList
#     """

#     def __init__(self, **kwargs):
#         super().__init__()

#     @abc.abstractmethod
#     def F(self, values) -> np.ndarray:
#         r"""The convective flux operator :math:`\pdeConvective`

#         Parameters
#         ----------
#         cells
#             A :class:`MeshCellSet` that contains the cell data

#         Returns
#         -------
#         F
#             An array of dimension :math:`Nx \times Ny \times N_\text{eqs}
#             \times N_\text{dim}` containing the computed convective flux for
#             the given problem
#         """
#         ...

#     @abc.abstractmethod
#     def B(self, values) -> np.ndarray:
#         r"""This returns the tensor that pre-multiplies the non-conservative
#         term of the problem.

#         A general problem can be written in a compact way:

#         .. math::

#             \pdeFull


#         This method needs to return :math:`\pdeNonConservativeMultiplier`

#         Parameters
#         ----------
#         state_array
#             The :class:`State` array containing the values of all the fields

#         Returns
#         -------
#         B
#             An array of dimension :math:`Nx \times Ny \times N_\text{eqs}
#             \times N_\text{state} \times N_\text{dim}` containing the computed
#             pre-multiplier tensor
#         """
#         pass

#     @abc.abstractmethod
#     def K(self, cells: Union[CellSet, MeshCellSet]) -> np.ndarray:
#         r"""This returns the tensor that pre-multiplies the gradient in the
#         diffusive term of the problem.

#         A general problem can be written in a compact way:

#         .. math::

#             \pdeFull


#         This method needs to return :math:`\pdeDiffusiveMultiplier`

#         Parameters
#         ----------
#         cells
#             A :class:`MeshCellSet` that contains the cell data

#         Returns
#         -------
#         D
#             An array of dimension :math:`Nx \times Ny \times N_\text{eqs}
#             \times N_\text{state} \times N_\text{dim} \times N_\text{dim}`
#             containing the computed pre-multiplier tensor
#         """
#         pass

#     @abc.abstractmethod
#     def s(self, values, t: float) -> np.ndarray:
#         r"""This returns the values of the source terms

#         A general problem can be written in a compact way:

#         .. math::

#             \pdeFull


#         This method needs to return :math:`\pdeSource`

#         Parameters
#         ----------
#         cells
#             A :class:`MeshCellSet` that contains the cell data

#         t
#             Time instant

#         Returns
#         -------
#         s
#             An array of dimension :math:`Nx \times Ny \times N_\text{eqs}`
#             containing the computed source terms
#         """
#         pass


class Problem(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super().__init__()


class ConvectiveProblem(Problem):
    @abc.abstractmethod
    def F(self, values) -> np.ndarray:
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


class NonConservativeProblem(Problem):
    @abc.abstractmethod
    def B(self, values) -> np.ndarray:
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


class DiffusiveProblem(Problem):
    @abc.abstractmethod
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


class SourceProblem(Problem):
    @abc.abstractmethod
    def s(self, values, t: float) -> np.ndarray:
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
