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
import numpy as np

from .state import Q


class Closure(metaclass=abc.ABCMeta):
    r""" A class representing the closure relation for :math:`p_I` and
    :math:`\vb{u}_I`. Use them as mixin with the Equation of State in order
    to provide full closure for the system
    """

    @abc.abstractmethod
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

    @abc.abstractmethod
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
        return state_array[..., Q.fields.p2]

    def uI(self, state_array: Q) -> np.ndarray:
        U1_V1 = state_array[..., Q.fields.U1 : Q.fields.V1 + 1]

        return U1_V1
