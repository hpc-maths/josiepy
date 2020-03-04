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
from __future__ import annotations

import copy
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Optional, NamedTuple, Union, TYPE_CHECKING

from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from .backend import PlotBackend

if TYPE_CHECKING:
    from josie.mesh import Mesh
    from josie.solver import Solver

# Default colorbar
# TODO: Make it configurable
cmap = plt.get_cmap("viridis")

StateData = Dict[str, np.ndarray]


class StateElement:
    """ An handy class to store the state of the plot at a given time
    """

    def __init__(self, time: float, data: StateData):
        self.time = time
        self.data = data


class PlotState:
    """ A class managing the global state of the plot backend

    Attributes
    ----------
    state_elements
        A list of :class:`StateElement` objects storing information about
        the artists to be plotted or animated at each time instant
    collection
        A global :class:`PatchCollection` used for all the plots
    """

    def __init__(self):
        self.collection: Optional[PatchCollection] = None
        self.state_elements: List[StateElement] = []

    def append(self, state_elem: StateElement):
        """ A proxy appending a state element to :attr:`state_elements`

        Parameters
        ---------
        state_elem
            The element to append
        """

        self.state_elements.append(state_elem)

    def __len__(self):
        return len(self.state_elements)

    def __getitem__(self, i):
        return self.state_elements[i]

    def __iter__(self):
        return iter(self.state_elements)


class AnimateState(NamedTuple):
    """ A named tuple to store state when calling the
    :meth:`MatplotlibBackend.show` method """

    fig: Figure
    animation: FuncAnimation


class MatplotlibBackend(PlotBackend):
    """ A :mod:`matplotlib`-based backend to plot 2D meshes.

    The state is stored in a list of :class:`StateElement` objects.

    """

    def __init__(self):
        self.plot_state: PlotState = PlotState()

    @staticmethod
    def _create_patch_collection(mesh: Mesh) -> PatchCollection:
        """ Create the patch collection exporting the patches data from
        the :class:`Mesh` object

        Parameters
        ----------
        mesh
            The :class:`Mesh` object from which to extract patches data

        Returns
        -------
        patch_coll
            The :class:`PatchCollection` object to be used then with
            :mod:`matplotlib` to actually visually display the mesh data
        """

        cells = []
        for i in range(mesh.num_cells_x):
            for j in range(mesh.num_cells_y):
                cells.append(Polygon(mesh.points[i, j, :, :]))

        return PatchCollection(cells, facecolors="white", edgecolors="k")

    def _init_solver_state(self, solver: Solver) -> StateData:
        """ Common work to do when plotting a solver state. It returns an
        updated :class:`StateData`
        """
        if self.plot_state.collection is None:
            # The mesh was not plotted before, we need to initialize our state
            self.plot(solver.mesh)

        # Otherwise we re-use the already created PatchCollection
        mesh_patch_coll: PatchCollection = self.plot_state.collection

        # Use the colormap
        mesh_patch_coll.set_cmap(cmap)

        # Add the data for each field
        state_data: StateData = {}
        for i, field in [
            (field.value, field.name) for field in solver.Q.fields
        ]:
            values = copy.deepcopy(solver.values[:, :, i].ravel())
            state_data[field] = values

        return state_data

    def plot(self, mesh: Mesh):
        patch_coll = self._create_patch_collection(mesh)

        self.plot_state.collection = patch_coll

    def update(self, solver: Solver):
        updated_state_data = self._init_solver_state(solver)

        if len(self.plot_state) == 0:
            self.plot_state.append(
                StateElement(time=0.0, data=updated_state_data)
            )
        else:
            self.plot_state[0].data = updated_state_data

    def append(self, solver: Solver, t):
        updated_state_data = self._init_solver_state(solver)

        state_element = StateElement(time=t, data=updated_state_data)

        self.plot_state.append(state_element)

    def _animate(self, fields: List[str], interval):
        """ Create an animation when more than one instant of time needs to be
        shown up

        Parameters
        ----------
        fields
            The list of fields to show. If `None`, then only the mesh is
            displayed
        interval
            Interval in ms between two frames in an animation
        """

        def ani_init():
            coll: PatchCollection = self.plot_state.collection
            coll.set_cmap(cmap)

            return coll

        def ani_step(state_element: StateElement, field: str):
            coll: PatchCollection = self.plot_state.collection
            data = state_element.data[field]

            vmin = np.min(data)
            vmax = np.max(data)

            coll.set_array(state_element.data[field])

            coll.set_clim(vmin, vmax)

            return coll

        ani_states: List[AnimateState] = []
        patch_coll = self.plot_state.collection
        for field in fields:
            fig: Figure
            ax: Axes

            fig, ax = plt.subplots()
            fig.colorbar(patch_coll)

            ax.add_collection(patch_coll)
            ax.autoscale_view()

            ani = FuncAnimation(
                fig,
                init_func=ani_init,
                repeat_delay=100,
                func=ani_step,
                frames=iter(self.plot_state),
                fargs=(field),
            )
            ani_states.append(AnimateState(fig=fig, animation=ani))

        plt.show()

    def show(self, fields: Union[List[str], str], fps=25):
        """ Show the plots on screen

        The logic to plot here is that if :attr:`plot_state` is a list
        with only 1 element, then single images are shown. Otherwise an
        animation is shown

        Parameters
        ----------
        fields
            The list of fields to show. If `None`, then only the mesh is
            displayed

        fps
            When animating, the number of frame per second [Default: 25]
        """

        t_instants = len(self.plot_state)

        # Force to 1-element list if just one string field is given
        if isinstance(fields, str):
            fields = [fields]

        if t_instants > 1:
            interval = t_instants / fps * 1000
            # We handle the animation in a separate method
            self._animate(fields, interval)
            return

        for field in fields:
            fig, ax = plt.subplots()

            # Here we have only one state_element ==> Single plot
            state_element: StateElement = self.plot_state[0]

            # Recover the patch collection from the state
            patch_coll: PatchCollection = copy.deepcopy(
                self.plot_state.collection
            )
            patch_coll.set_array(state_element.data[field])
            patch_coll.set_cmap(cmap)
            ax.add_collection(patch_coll)

        plt.show()

    def show_grid(self):
        fig, ax = plt.subplots()

        patch_coll = self.plot_state.collection
        ax.add_collection(patch_coll)
        plt.show()

    def show_all(self):
        fields = self.plot_state.keys()
        self.show(fields)
