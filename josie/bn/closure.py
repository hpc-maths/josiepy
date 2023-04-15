# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
import numpy as np

from .state import Q


class Closure(ABC):
    r"""A class representing the closure relation for :math:`p_I` and
    :math:`\vb{u}_I`. Use them as mixin with the Equation of State in order
    to provide full closure for the system
    """

    @abstractmethod
    def pI(self, state_array: Q) -> np.ndarray:
        r"""
        Parameters
        ----------
        Q
            A :math:`Nx \times Ny \times 9` array containing the values for
            all the state variables

        Returns
        -------
        pI
            A :math:`Nx \times Ny \times 1` array containing the value of the
            :math:`p_I`
        """

        raise NotImplementedError

    @abstractmethod
    def uI(self, state_array: Q) -> np.ndarray:
        r"""
        Parameters
        ---------
        Q
            A :math:`Nx \times Ny \times 9` array containing the values for
            all the state variables

        Returns
        ------
        uI
            A :math:`Nx \times Ny \times 2` array that contains the components
            of the velocity :math:`\vb{u}_I`.
        """
        raise NotImplementedError


class Classical(Closure):
    r""" This is the classical choice for :math:`p_I` and :math:`\vb{u}_I`
    described in :cite:`baer_two-phase_1986`

    .. math::

        p_I = p_2 \\
        \vb{u}_I = \vb{u}_1
    """

    def pI(self, state_array: Q) -> np.ndarray:
        return state_array[..., [Q.fields.p2]]

    def uI(self, state_array: Q) -> np.ndarray:
        U1_V1 = state_array[..., [Q.fields.U1, Q.fields.V1]]

        return U1_V1
