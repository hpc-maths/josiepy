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
""" This module contains the different Equation of State (EOS) implementations
"""

import abc
import numpy as np


class EOS(metaclass=abc.ABCMeta):
    """ An Abstract Base Class representing an EOS for an Euler System """

    @abc.abstractmethod
    def rhoe(self, rho: np.ndarray, p: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def p(self, rho: np.ndarray, e: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def sound_velocity(self, rho: np.ndarray, p: np.ndarray):
        raise NotImplementedError


class PerfectGas(EOS):
    r""" This class embeds methods to compute states for the Euler problem
    using an EOS (Equation of State) for perfect gases

    .. math::

        p = \rho \mathcal{R} T = \rho \left( \gamma - 1 \right)e


    Attributes
    ----------
    gamma
        The adiabatic coefficient
    """

    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    def rhoe(self, rho: np.ndarray, p: np.ndarray):
        """ This returns the internal energy multiplied by the density

        Parameters
        ----------
        rho
            A :class:`np.ndarray` containing the values of the density on the
            mesh cells

        p
            A :class:`np.ndarray` containing the values of the pressure on the
            mesh cells

        Returns
        -------
        rhoe
            A :class`np.ndarray` containing the values of the internal energy
            multiplied by the density
        """

        return p / (self.gamma - 1)

    def p(self, rho: np.ndarray, e: np.ndarray):
        """ This returns the pressure from density and internal energy

        Parameters
        ----------
        rho
            A :class:`np.ndarray` containing the values of the density on the
            mesh cells

        e
            A :class:`np.ndarray` containing the values of the internal energy
            on the mesh cells

        Returns
        -------
        p
            A :class`np.ndarray  containing the values of the pressure
            multiplied by the density
        """
        return (self.gamma - 1) * np.multiply(rho, e)

    def sound_velocity(self, rho: np.ndarray, p: np.ndarray):
        """ This returns the sound velocity from density and pressure

        Parameters
        ----------
        rho
            A :class:`np.ndarray` containing the values of the density on the
            mesh cells

        p
            A :class:`np.ndarray` containing the values of the pressure
            on the mesh cells

        Returns
        -------
        c
            A :class`np.ndarray  containing the values of the sound velocity
            multiplied by the density
        """

        return np.sqrt(self.gamma * np.divide(p, rho))
