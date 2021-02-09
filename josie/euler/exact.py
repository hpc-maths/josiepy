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
from __future__ import annotations

import numpy as np

from enum import auto, Enum, IntEnum
from typing import Type, Tuple
from scipy.optimize import root, root_scalar
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from josie.euler.eos import EOS
from josie.euler.state import Q
from josie.state import Fields, State


class Wave(IntEnum):
    LEFT = 1
    RIGHT = -1


class WaveType(Enum):
    SHOCK = auto()
    RAREFACTION = auto()


class RarefactionFields(Fields):
    U = 0
    rho = 1


class RarefactionState(State):
    fields: Type[RarefactionFields]


class Exact:
    r"""This class implements the exact solution of the Riemann problem as
    scheme.

    See :cite:`toro_riemann_2009` for a detailed view on compressible schemes
    and :cite:`godlewski_numerical_1996` for a deep study on hyperbolic systems
    and the Riemann problem. This is also based on
    :cite:`colella_efficient_1985` and :cite:`kamm_exact_2015`

    Parameters
    ----------
    eos
        The :class:`EOS` to use

    Q_L
        The left state

    Q_R The right state

    Attributes
    ----------
    eos
        The :class:`EOS` to use

    Q_L
        The left state

    Q_R The right state

    Q_star_L
        The left star state

    Q_star_R
        The right star state

    left_wave
        The type of the left wave (it's populated after calling :meth:`solve`

    right_wave
        The type of the right wave (it's populated after calling :meth:`solve`

    """
    _rho0: float
    _interpolators: dict
    left_wave: WaveType
    right_wave: WaveType

    def __init__(self, eos, Q_L: Q, Q_R: Q):
        self.eos = eos
        self.Q_L = Q_L
        self.Q_R = Q_R

        self._interpolators = {}

    def _set_state(self, rho: float, p: float, U: float, V: float) -> Q:
        """Handy function to set a full :class:`Q` state from density,
        pressure, and velocity"""
        fields = self.Q_L.fields
        state = np.empty_like(self.Q_L)

        state[..., fields.rho] = rho
        state[..., fields.rhoU] = rho * U
        state[..., fields.rhoV] = rho * V
        rhoe = state[..., fields.rhoe] = self.eos.rhoe(rho, p)
        state[..., fields.rhoE] = rho * (rhoe / rho + U ** 2 / 2)
        state[..., fields.c] = self.eos.sound_velocity(rhoe, p)

        state[..., fields.U] = U
        state[..., fields.c] = self.eos.sound_velocity(rho, p)
        state[..., fields.V] = V
        state[..., fields.p] = p

        return state

    def rankine_hugoniot(self, rho: float, p: float, rho_k: float, p_k: float):
        """The Rankine-Hugoniot locus connecting the known state before the
        :math:`k` shock wave to the "star" state after the shock wave"""

        eos: EOS = self.eos

        tau = 1 / rho
        tau_k = 1 / rho_k

        e = eos.rhoe(rho, p) / rho
        e_k = eos.rhoe(rho_k, p_k) / rho_k

        return e - e_k + 0.5 * (p + p_k) * (tau - tau_k)

    def _solve_rankine_hugoniot(self, p: float, rho_k: float, p_k: float):
        r"""Solve the non-linear Rankine-Hugoniot given a constant :math:`p^*`
        and a :math:`\rho_0` first guess. :math:`\rho_k` and :math:`p_k` are
        parameters
        """

        # Solve the R-H condition
        opt = root(
            self.rankine_hugoniot,
            self._rho0,
            args=(p, rho_k, p_k),
        )

        if not (opt.success):
            raise RuntimeError(
                "The root finding algorithm could not find a solution for "
                "rho_star_k"
            )

        rho_star_k = opt.x

        return rho_star_k

    def shock(self, p: float, Q_k: Q, wave: Wave) -> float:
        """This function returns the after-shock velocity

        Parameters
        ----------
        p
            The after shock pressure value

        rho_guess
            An initial estimate for the after shock density

        Q_k
            The state before the shock

        wave
            The wave to consider

        Returns
        -------
        The value of the non linear function after the shock
        """
        fields = Q_k.fields
        rho_k = Q_k[..., fields.rho]
        tau_k = 1 / rho_k
        p_k = Q_k[..., fields.p].item()

        rho_star_k = self._solve_rankine_hugoniot(p, rho_k, p_k)

        self._rho0 = rho_star_k

        tau_star_k = 1 / rho_star_k

        return (p - p_k) * np.sqrt((tau_k - tau_star_k) / (p - p_k))

    def rarefaction_ode(self, p, y, wave: Wave):
        """Solve the rarefaction ODE for a vectorized set of states

        Parameters
        ----------
        p
            is the indipendent variable, i.e. the pressure

        q
            is the state variable. It's an array of size [num_riemann x 2].

        Returns
        -------
        derivatives
            A :class:`tuple` containing the derivatives of the state
        """
        eos: EOS = self.eos
        # u = y[0]
        rho = y[1]
        c = eos.sound_velocity(rho, p)

        drho_dp = 1 / c ** 2
        du_dp = wave / rho / c

        return (du_dp, drho_dp)

    def _solve_rarefaction_ode(
        self,
        p_span: Tuple[float, float],
        initial_state: Tuple[float, float],
        wave: Wave,
    ):
        """This solves the rarefaction ODE from ``p_span[0] to p_span[1]``
        given ``initial_state`` for the provide :class:`Wave`.

        The first element of the ``initial_state`` is the velocity, the second
        is the density"""

        def rhs(p, y):
            return self.rarefaction_ode(p, y, wave)

        # Let's use a fine enough dp that helps us after for interpolation
        num_steps = 100

        max_step = (p_span[0] - p_span[1]) / num_steps

        sol = solve_ivp(
            rhs, p_span, initial_state, first_step=max_step, max_step=max_step
        )

        return sol

    def _solve_rarefaction(
        self,
        p_star: float,
        p_k: float,
        U_k: float,
        V_k: float,
        rho_k: float,
        wave: Wave,
    ) -> float:
        """This handy method creates the cubic interpolators after solving the
        rarefaction ODE and returns the :math:`rho^*_k` associated to the given
        :math:`p_k` to :math:`p^*`

        Returns
        -------
        rho_star_k
            THe value of the density after the rarefaction fan
        """

        # Solve rarefaction ODE
        ode_sol = self._solve_rarefaction_ode(
            (p_k, p_star), (U_k, rho_k), Wave.LEFT
        )

        # Accumulate the full state for the rarefaction
        rar_full_state = np.empty((len(ode_sol.t), len(Q.fields))).view(Q)

        for i, p_step in enumerate(ode_sol.t):
            rho_step = ode_sol.y[1, i]
            U_step = U_k - wave * (ode_sol.y[0, i] - U_k)

            rar_full_state[i, :] = self._set_state(
                rho_step, p_step, U_step, V_k
            )

        U = rar_full_state[..., Q.fields.U]
        rho = rar_full_state[..., Q.fields.rho]
        p = rar_full_state[..., Q.fields.p]
        c = rar_full_state[..., Q.fields.c]

        # Create cubic interpolators
        if wave is Wave.LEFT:
            self._interpolators[Wave.LEFT.name] = {
                "p": interp1d(U - c, p, kind="cubic"),
                "rho": interp1d(U - c, rho, kind="cubic"),
                "U": interp1d(U - c, U, kind="cubic"),
            }

        else:
            self._interpolators[Wave.RIGHT.name] = {
                "p": interp1d(U + c, p, kind="cubic"),
                "rho": interp1d(U + c, rho, kind="cubic"),
                "U": interp1d(U + c, U, kind="cubic"),
            }

        return rho[-1]

    def rarefaction(self, p: float, Q_k: Q, wave: Wave) -> float:
        r"""Non linear function for the rarefaction

        Parameters
        ----------
        p
            The after rarefaction fan pressure value

        rho
            An initial estimate for the after rarefaction density (unused)

        Q_k
            The state before the rarefaction fan

        wave
            The wave to consider

        Returns
        -------
        The value of the non linear function :math:`f_k(p^*)` which statisfies
        :math:`u^* = U_k \pm f_k(p^*)`
        """
        fields = Q_k.fields
        rho_k = Q_k[..., fields.rho]
        p_k = Q_k[..., fields.p]
        U_k = Q_k[..., fields.U]

        ode_sol = self._solve_rarefaction_ode((p_k, p), (U_k, rho_k), wave)

        # Last value corresponds to the star region value
        U = ode_sol.y[0, -1]

        return wave * (U - U_k)

    def sample_rarefaction(self, U_c: float, V_k: float, wave: Wave) -> Q:
        r""" Return the state within the rarefaction fan """

        if wave is Wave.LEFT:
            interps = self._interpolators[Wave.LEFT.name]
            p = interps["p"](U_c)
            rho = interps["rho"](U_c)
            U = interps["U"](U_c)
            return self._set_state(rho, p, U, V_k)
        else:
            interps = self._interpolators[Wave.RIGHT.name]
            p = interps["p"](U_c)
            rho = interps["rho"](U_c)
            U = interps["U"](U_c)
            return self._set_state(rho, p, U, V_k)

    def solve(self):
        fields = self.Q_L.fields
        Q_L = self.Q_L
        Q_R = self.Q_R

        p_L = Q_L[..., fields.p]
        rho_L = Q_L[..., fields.rho]
        U_L = Q_L[..., fields.U]

        p_R = Q_R[..., fields.p]
        rho_R = Q_R[..., fields.rho]
        U_R = Q_R[..., fields.U]

        def f_L(p):
            """ The velocity after the left wave """
            if p > p_L:
                return self.shock(p, Q_L, Wave.LEFT)
            else:
                return self.rarefaction(p, Q_L, Wave.LEFT)

        def f_R(p):
            """ The velocity after the right wave """
            if p > p_R:
                return self.shock(p, Q_R, Wave.RIGHT)
            else:
                return self.rarefaction(p, Q_R, Wave.RIGHT)

        def f(p):
            return f_R(p) + f_L(p) + (U_R - U_L)

        # First Guess rho
        self._rho0 = 0.5 * (rho_L + rho_R)

        opt = root_scalar(f, bracket=[0, 1e8])

        if not (opt.converged):
            raise RuntimeError(
                "The root finding algorithm could not find a solution for "
                "p_star"
            )

        p_star = opt.root

        # Final estimate of _star region velocity
        U_star = 0.5 * ((U_L + U_R) + (f_R(p_star) - f_L(p_star)))

        return self.full_state(p_star, U_star)

    def full_state(self, p_star: float, U_star: float):
        """Compute the full Euler state from the value of :math:`p^*` and
        :math:`u^*`
        """
        fields = self.Q_L.fields
        Q_L = self.Q_L
        Q_R = self.Q_R

        p_L = Q_L[..., fields.p]
        rho_L = Q_L[..., fields.rho]
        U_L = Q_L[..., fields.U]
        V_L = Q_L[..., fields.V]

        p_R = Q_R[..., fields.p]
        rho_R = Q_R[..., fields.rho]
        U_R = Q_R[..., fields.U]
        V_R = Q_R[..., fields.V]

        # Left Star State
        if p_star > p_L:
            self.left_wave = WaveType.SHOCK

            rho_star_L = self._solve_rankine_hugoniot(p_star, rho_L, p_L)
            Q_star_L = self._set_state(rho_star_L, p_star, U_star, V_L)

            # Shock speed
            self.left_shock_speed = (rho_L * U_L - rho_star_L * U_star) / (
                rho_L - rho_star_L
            )
        else:
            self.left_wave = WaveType.RAREFACTION

            # Retrieve rarefaction full state
            rho_star_L = self._solve_rarefaction(
                p_star, p_L, U_L, V_L, rho_L, Wave.LEFT
            )

            Q_star_L = self._set_state(rho_star_L, p_star, U_star, V_L)

        # Right Star State
        if p_star > p_R:
            self.right_wave = WaveType.SHOCK

            rho_star_R = self._solve_rankine_hugoniot(p_star, rho_R, p_R)
            Q_star_R = self._set_state(rho_star_R, p_star, U_star, V_R)

            # Shock speed
            self.right_shock_speed = (rho_R * U_R - rho_star_R * U_star) / (
                rho_R - rho_star_R
            )

        else:
            self.right_wave = WaveType.RAREFACTION

            # Retrieve rarefaction full state
            rho_star_R = self._solve_rarefaction(
                p_star, p_R, U_R, V_R, rho_R, Wave.RIGHT
            )

            Q_star_R = self._set_state(rho_star_R, p_star, U_star, V_R)

        self.Q_star_L = Q_star_L
        self.Q_star_R = Q_star_R

    def sample(self, x: float, t: float, origin: float = 0.5) -> Q:
        """Sample the solution at given :math:`(x, t)`

        Returns
        -------
        state
            The state solution at the requested :math:`(x, t)`
        """
        fields = self.Q_star_L.fields
        U_star = self.Q_star_L[..., fields.U]

        speed = (x - origin) / t

        if speed < U_star:
            if self.left_wave is WaveType.SHOCK:
                # State is at the left of the shockwave
                if speed <= self.left_shock_speed:
                    return self.Q_L

                # State is in the left-star zone
                if speed > self.left_shock_speed and speed <= U_star:
                    return self.Q_star_L
            else:
                U_L = self.Q_L[..., fields.U]
                V_L = self.Q_L[..., fields.V]
                c_L = self.Q_L[..., fields.c]

                c_star_L = self.Q_star_L[..., fields.c]

                rarefaction_head = U_L - c_L
                rarefaction_tail = U_star - c_star_L

                # States at the left of the rarefaction fan
                if speed < rarefaction_head:
                    return self.Q_L

                # State at the right of the rarefaction fan
                if speed > rarefaction_tail:
                    return self.Q_star_L

                # States within the rarefaction fan. We interpolate
                return self.sample_rarefaction(speed, V_L, Wave.LEFT)
        else:  # > U_star
            if self.right_wave is WaveType.SHOCK:
                # State is at the right of the shockwave
                if speed >= self.right_shock_speed:
                    return self.Q_R

                # State is in the right-star zone
                if speed < self.right_shock_speed and speed >= U_star:
                    return self.Q_star_R
            else:
                U_R = self.Q_R[..., fields.U]
                V_R = self.Q_R[..., fields.V]
                c_R = self.Q_R[..., fields.c]

                c_star_R = self.Q_star_R[..., fields.c]

                rarefaction_head = U_R + c_R
                rarefaction_tail = U_star + c_star_R

                # States at the right of the rarefaction fan
                if speed > rarefaction_head:
                    return self.Q_R

                # State at the left of the rarefaction fan
                if speed < rarefaction_tail:
                    return self.Q_star_R

                return self.sample_rarefaction(speed, V_R, Wave.RIGHT)

        # Never reached (Added for type check)
        assert False
        return None
