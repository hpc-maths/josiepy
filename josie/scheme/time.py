# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import abc

from josie.mesh import Mesh
from josie.problem import Problem

from .scheme import Scheme


class TimeScheme(Scheme):
    r"""A mixin that provides the scheme implementation for the time
    derivative

    .. math::

        \numTime

    Attributes
    ----------
    order
        The supposed order for the scheme. Useful for exemple when testing.
    """

    time_order: float

    def __init__(self, problem: Problem, *args, **kwargs):
        super().__init__(problem)

    @abc.abstractmethod
    def step(self, mesh: Mesh, dt: float, t: float):
        raise NotImplementedError
