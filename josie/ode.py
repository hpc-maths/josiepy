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
""" Handy objects to speed up the setup of a simulation for a problem governed
by an Ordinary Differential Equation (ODE) (i.e. time-varying only)
"""
from typing import Callable, Type

from josie.boundary.set import UnitCube
from josie.general.schemes.source import ConstantSource
from josie.mesh import Mesh
from josie.mesh.cell import SimpleCell
from josie.mesh.cellset import MeshCellSet
from josie.solver import Solver
from josie.solver.problem import Problem
from josie.solver.state import State
from josie.solver.scheme.time import TimeScheme


class OdeProblem(Problem):
    def __init__(self, f: Callable[[State], State]):
        self.f = f

    def s(self, cells: MeshCellSet) -> State:
        return -self.f(cells.values)


class OdeSolver(Solver):
    """A convenience solver that takes care of all the required setups of
    objects to speed up the configuration of an ODE problem

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(
        self,
        Q: Type[State],
        Scheme: Type[TimeScheme],
        f: Callable[[State], State],
        *args,
        **kwargs
    ):
        domain = UnitCube()

        # Set None Bc to create a 0D case
        for boundary in domain:
            boundary.bc = None

        mesh = Mesh(
            domain.left, domain.bottom, domain.right, domain.top, SimpleCell
        )
        mesh.interpolate(1, 1)
        mesh.generate()

        Scheme = self._wrap_time_scheme(Scheme, f)

        scheme = Scheme(OdeProblem(f), *args, **kwargs)

        super().__init__(mesh, Q, scheme)

    @staticmethod
    def _wrap_time_scheme(
        Scheme: Type[TimeScheme], f: Callable[[State], State], *args, **kwargs
    ):
        """This method wraps the provided time scheme in order to prevent
        exceptions due to abstract methods that are not used in a ODE context

        Parameters
        ----------
        scheme
            The :class:`TimeScheme` to wrap

        args
            You can provide additional parameters that need to be passed to the
            :meth:`TimeScheme.__init__`
        """

        class _WrappedScheme(ConstantSource, Scheme):  # type: ignore
            # CFL is not needed for ODE
            def CFL(self):
                pass

        _WrappedScheme.__name__ = Scheme.__name__

        return _WrappedScheme
