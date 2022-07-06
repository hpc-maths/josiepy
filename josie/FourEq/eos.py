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

from josie.twofluid.state import PhasePair
import abc
import numpy as np


from typing import Union

ArrayAndScalar = Union[np.ndarray, float]


class BarotropicEOS(metaclass=abc.ABCMeta):
    """An Abstract Base Class representing a barotropic EOS for an
    Euler System"""

    @abc.abstractmethod
    def p(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def rho(self, p: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abc.abstractmethod
    def sound_velocity(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError


class LinearizedGas(BarotropicEOS):
    r"""This class embeds methods to compute states for the Euler problem
    using an EOS (Equation of State) for linearized gases

    .. math::

        p = p_0 + c_0^2(\rho - \rho_0)


    Attributes
    ----------
    p_0
        The pressure of reference
    rho_0
        The density of reference
    c_0
        The sound velocity of reference
    """

    def __init__(self, p0: float, rho0: float, c0: float):
        self.p0 = p0
        self.rho0 = rho0
        self.c0 = c0

    def p(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the pressure from density

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells


        Returns
        -------
        p
            A :class:`ArrayAndScalar  containing the values of the pressure
            on the mesh cells
        """
        return self.p0 + self.c0 * self.c0 * (rho - self.rho0)

    def rho(self, p: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the density from pressure

        Parameters
        ----------
        p
            A :class:`ArrayAndScalar` containing the values of the pressure
            on the mesh cells

        Returns
        -------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells
        """

        return self.rho0 + (p - self.p0) / (self.c0 * self.c0)

    def sound_velocity(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the frozen speed of sound from density

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        Returns
        -------
        c
            A :class:`ArrayAndScalar` containing the values of the sound
            velocity multiplied by the density
        """

        return self.c0


class PolytropicGas(BarotropicEOS):
    r"""This class embeds methods to compute states for the Euler problem
    using an EOS (Equation of State) for polytropic gases

    .. math::

        p = K \rho^{\gamma}


    Attributes
    ----------
    gamma
        The polytropic exponent
    K
        The polytropic constant
    """

    def __init__(self, gamma: float = 1.4, K: float = 0.12):
        self.gamma = gamma
        self.K = K

    def p(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the pressure from density

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells


        Returns
        -------
        p
            A :class:`ArrayAndScalar  containing the values of the pressure
            on the mesh cells
        """
        return self.K * rho ** self.gamma

    def rho(self, p: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the density from pressure

        Parameters
        ----------
        p
            A :class:`ArrayAndScalar` containing the values of the pressure
            on the mesh cells

        Returns
        -------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells
        """

        return p / self.K ** (1.0 / self.gamma)

    def sound_velocity(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the frozen speed of sound from density

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        Returns
        -------
        c
            A :class:`ArrayAndScalar` containing the values of the sound
            velocity multiplied by the density
        """

        return np.sqrt(self.K * self.gamma * rho ** (self.gamma - 1.0))


class TaitEOS(BarotropicEOS):
    r"""This class embeds methods to compute states for the Euler problem
    using an EOS (Equation of State) of Tait type

    .. math::

        p = \frac{\rho_{0}c_{0}^{2}}{n_{0}}
            \left[\left(\frac{\rho}{\rho_{0}}\right)^{n_{0}} + 1\right] + p_{0}


    Attributes
    ----------
    rho0
        The reference density
    p0
        The reference pressure
    c0
        The reference speed of sound
    n0
        The Tait exponent
    """

    def __init__(
        self,
        rho0: float = 1000,
        p0: float = 3450,
        c0: float = 1500,
        n0: float = 7.15,
    ):
        self.rho0 = rho0
        self.p0 = p0
        self.c0 = c0
        self.n0 = n0

    def p(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the pressure from density

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells


        Returns
        -------
        p
            A :class:`ArrayAndScalar  containing the values of the pressure
            multiplied by the density
        """
        return (
            self.rho0
            * self.c0
            * self.c0
            / self.n0
            * ((rho / self.rho0) ** (self.n0) - 1.0)
            + self.p0
        )

    def rho(self, p: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the density from pressure

        Parameters
        ----------
        p
            A :class:`ArrayAndScalar` containing the values of the pressure
            on the mesh cells

        Returns
        -------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells
        """

        return self.rho0 * np.power(
            (p - self.p0) * self.n0 / (self.rho0 * self.c0 * self.c0) + 1.0,
            1.0 / self.n0,
        )

    def sound_velocity(self, rho: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the frozen speed of sound from density

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        Returns
        -------
        c
            A :class:`ArrayAndScalar` containing the values of the sound
            velocity multiplied by the density
        """

        return np.sqrt(
            self.c0 * self.c0 * (rho / self.rho0) ** (self.n0 - 1.0)
        )


class TwoPhaseEOS(PhasePair):
    """An Abstract Base Class representing en EOS for a twophase system.  In
    particular two :class:`.euler.eos.EOS` instances for each phase need to be
    provided.

    You can access the EOS for a specified phase using the
    :meth:`__getitem__`

    """

    def __init__(self, phase1: BarotropicEOS, phase2: BarotropicEOS):
        """
        Parameters
        ----------
        phase1
            An instance of :class:`.euler.eos.EOS` representing the EOS for the
            single phase #1
        phase2
            An instance of :class:`.euler.eos.EOS` representing the EOS for the
            single phase #2
        """

        super().__init__(phase1, phase2)
