# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" Classes associated to the implementation of Runge-Kutta time schemes """
from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import TYPE_CHECKING


from josie.mesh import Mesh
from josie.mesh.cellset import MeshCellSet
from josie.scheme.time import TimeScheme

if TYPE_CHECKING:
    from josie.problem import Problem


@dataclass
class ButcherTableau:
    r""" A class to store the RK coefficients.

    Generally coefficients for a generic Runge-Kutta method are stored in a
    mnemonic structure called Butcher Tableau.

    .. math::

        \renewcommand\arraystretch{1.2}
        \begin{array}
        {c|cccc}
        0\\
        c_1 & a_{11} \\
        c_2 & a_{12} & a_{22} \\
        \vdots & \ldots \\
        c_s & a_{s1} & a_{s2} & \ldots & a_{s s-1} \\
        \hline
        & b_1 & b_2 & \ldots & b_s
        \end{array}


    For example:

    .. math::

        \renewcommand\arraystretch{1.2}
        \begin{array}
        {c|cccc}
        0\\
        \frac{1}{2} & \frac{1}{2}\\
        \frac{1}{2} &0 &\frac{1}{2} \\
        1& 0& 0& 1\\
        \hline
        & \frac{1}{6} &\frac{1}{3} &\frac{1}{3} &\frac{1}{6}
        \end{array}

    Parameters
    ----------
    a_s
        The :math:`a_s` coefficients stored in a :class:`np.ndarray`. The order
        of storage is :math:`a_{11}, a_{21}, a_{22}, a_{31} \ldots`
    b_s
        The :math:`b_s` coefficients stored in a :class:`np.ndarray`. The order
        of storage is :math:`b_1, b_2, b_3, b_4 \ldots`
    c_s
        The :math:`c_s` coefficients stored in a :class:`np.ndarray`. The order
        of storage is :math:`c_1, c_2, c_3` \ldots`


    Attributes
    ----------

    a_s
        The :math:`a_s` coefficients stored in a :class:`np.ndarray`. The order
        of storage is :math:`a_{11}, a_{21}, a_{22}, a_{31} \ldots`
    b_s
        The :math:`b_s` coefficients stored in a :class:`np.ndarray`. The order
        of storage is :math:`b_1, b_2, b_3, b_4 \ldots`
    c_s
        The :math:`c_s` coefficients stored in a :class:`np.ndarray`. The order
        of storage is :math:`c_1, c_2, c_3 \ldots`
    """

    a_s: np.ndarray
    b_s: np.ndarray
    c_s: np.ndarray


class RK(TimeScheme):
    r"""A generic Runge-Kutta explicit method

    .. math::

        \rungeKutta

    Attributes
    ----------
    problem
        An instance of :class:`Problem` representing the physical problem that
        this scheme discretizes

    butcher
        An instance of :class:`ButcherTableau` that provides the coefficient of
        the Runge-Kutta method

    """

    def __init__(self, problem: Problem, butcher: ButcherTableau):
        # To make if work with the recursive :math:`k` function, we need to add
        # an initial value of 0 for c_s and a_s

        super().__init__(problem)

        butcher.c_s = np.insert(butcher.c_s, 0, 0)
        butcher.a_s = np.insert(butcher.a_s, 0, 0)

        # Error checking for coefficients
        if not (len(butcher.c_s) == len(butcher.b_s)):
            raise ValueError(
                "The number of `c_s` coefficients must be " "the same as the `b_s`"
            )

        self.butcher = butcher
        self.num_steps: int = len(butcher.b_s)

    def post_init(self, cells: MeshCellSet):
        r"""A Runge-Kutta method needs to store intermediate steps. It needs
        :math:`s - 1` additional storage slots, where :math:`s` is the number
        of steps of the Runge-Kutta method

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells
        """

        super().post_init(cells)

        nx, ny, num_dofs, num_fields = cells.values.shape

        self._ks: np.ndarray = np.empty(
            (nx, ny, num_dofs, num_fields, self.num_steps - 1)
        )

    def pre_step(self, cells: MeshCellSet, dt: float):
        """Zero-out the array containing the :math:`k_s` values

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells
        """

        super().pre_step(cells, dt)

        self._ks.fill(0)

    def k(self, mesh: Mesh, dt: float, t: float, step: int):
        r"""Recursive function that computes all the :math:`k_s` coefficients
        from :math:`s = 0` to :math:`s = \text{step}`

        The highest :math:`k_s` value is stored in :attr:`_fluxes`
        """

        if step > 0:
            self.k(mesh, dt, t, step - 1)
            self._ks[..., step - 1] = self._fluxes.copy()
            self._fluxes.fill(0)

        c = self.butcher.c_s[step]
        a_s = self.butcher.a_s[step : 2 * step + 1]

        t += c * dt
        step_cells = mesh.cells.copy()
        step_cells.values = (
            step_cells.values
            - dt
            * np.einsum("k,...k->...", a_s, self._ks[..., :step])
            / mesh.cells.volumes[..., np.newaxis, np.newaxis]
        )
        step_cells.update_ghosts(mesh.boundaries, t)

        self.pre_accumulate(step_cells, dt, t)

        for neighs in step_cells.neighbours:
            self.accumulate(step_cells, neighs, t)

    def step(self, mesh: Mesh, dt: float, t: float):
        self.k(mesh, dt, t, self.num_steps - 1)
        # Now self._fluxes contains the last k value. So we multiply the
        # corresponding b
        self._fluxes = self._fluxes * self.butcher.b_s[-1]

        # Let's sum all the other contributions from 0 to s-1
        self._fluxes = self._fluxes + np.einsum(
            "i,...i->...", self.butcher.b_s[:-1], self._ks
        )


class RK2Alpha(RK):
    r"""Implements the explicit 2nd-order Runge-Kutta scheme with a tunable
    :math:`\alpha` parameter

    .. math::

        \pdeState^{k+1} = \pdeState^k +
            \Delta t \;
            \vb{f}\qty(\qty(1-\frac{1}{2\alpha})k_1 + \frac{1}{2\alpha})


    .. math::

        \renewcommand\arraystretch{1.2}
        \begin{array}
        {c|cccc}
        0\\
        \alpha & \alpha \\
        \hline
        & \qty(1 - \frac{1}{2\alpha}) & \frac{1}{2\alpha}
        \end{array}

    Parameters
    ----------
    problem
        An instance of :class:`Problem` representing the physical problem that
        this scheme discretizes

    alpha
        The value of the alpha coefficient

    """

    def __init__(self, problem: Problem, alpha: float):
        self.alpha = alpha

        butcher = ButcherTableau(
            a_s=np.array([alpha]),
            b_s=np.array([1 - 1 / (2 * alpha), 1 / (2 * alpha)]),
            c_s=np.array([alpha]),
        )

        super().__init__(problem, butcher)


class RK2(RK2Alpha):
    r"""Implements the explicit 2nd-order Runge-Kutta scheme with :math:`\alpha =
    2/3`
    """

    time_order: float = 2

    def __init__(self, problem: Problem):
        super().__init__(problem, 2 / 3)
