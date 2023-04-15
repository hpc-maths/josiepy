# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" This module contains the different Equation of State (EOS) implementations
"""

from abc import ABC, abstractmethod
import numpy as np

from typing import Union

ArrayAndScalar = Union[np.ndarray, float]


class EOS(ABC):
    """An Abstract Base Class representing an EOS for an Euler System"""

    @abstractmethod
    def rhoe(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abstractmethod
    def p(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abstractmethod
    def rho(self, p: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        raise NotImplementedError

    @abstractmethod
    def sound_velocity(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        raise NotImplementedError


class PerfectGas(EOS):
    r"""This class embeds methods to compute states for the Euler problem
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

    def rhoe(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the internal energy multiplied by the density

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        p
            A :class:`ArrayAndScalar` containing the values of the pressure on
            the mesh cells

        Returns
        -------
        rhoe
            A :class:`ArrayAndScalar` containing the values of the internal
            energy multiplied by the density
        """

        return p / (self.gamma - 1)

    def p(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the pressure from density and internal energy

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        e
            A :class:`ArrayAndScalar` containing the values of the internal
            energy on the mesh cells

        Returns
        -------
        p
            A :class:`ArrayAndScalar  containing the values of the pressure
            multiplied by the density
        """
        return (self.gamma - 1) * np.multiply(rho, e)

    def rho(self, p: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        """This returns the density from pressure and internal energy

        Parameters
        ----------
        p
            A :class:`ArrayAndScalar` containing the values of the pressure
            on the mesh cells

        e
            A :class:`ArrayAndScalar` containing the values of the internal
            energy on the mesh cells


        Returns
        -------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells
        """

        return p / (self.gamma - 1) / e

    def sound_velocity(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        """This returns the sound velocity from density and pressure

        Parameters
        ----------
        rho
            A :class:`ArrayAndScalar` containing the values of the density on
            the mesh cells

        p
            A :class:`ArrayAndScalar` containing the values of the pressure
            on the mesh cells

        Returns
        -------
        c
            A :class:`ArrayAndScalar` containing the values of the sound
            velocity multiplied by the density
        """

        return np.sqrt(self.gamma * np.divide(p, rho))


class StiffenedGas(PerfectGas):
    def __init__(self, gamma: float = 3, p0: float = 1e9):
        self.gamma = gamma
        self.p0 = p0

    def rhoe(self, rho: ArrayAndScalar, p: ArrayAndScalar) -> ArrayAndScalar:

        return (p + self.gamma * self.p0) / (self.gamma - 1)

    def p(self, rho: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:

        return (self.gamma - 1) * np.multiply(rho, e) - self.p0 * self.gamma

    def rho(self, p: ArrayAndScalar, e: ArrayAndScalar) -> ArrayAndScalar:
        return (p + self.p0 * self.gamma) / (self.gamma - 1) / e

    def sound_velocity(
        self, rho: ArrayAndScalar, p: ArrayAndScalar
    ) -> ArrayAndScalar:
        return np.sqrt(self.gamma * np.divide((p + self.p0), rho))
