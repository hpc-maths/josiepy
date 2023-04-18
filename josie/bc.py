# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import numpy as np

from typing import Callable, Tuple, TYPE_CHECKING, Union

from josie.state import State

from .boundary import Boundary, BoundaryCurve
from .data import NoAliasEnum
from .math import Direction

if TYPE_CHECKING:
    from josie.mesh.cellset import CellSet, MeshCellSet

    Number = Union[int, float]
    BCCallable = Callable[[CellSet, float], np.ndarray]
    ImposedValue = Union[BCCallable, State, Number]


class SetValueCallable:
    """Used to convert the bare float value to a callable returning constant
    value for all the cells. Used for :class:`Dirichlet` and children classes
    """

    def __init__(self, value: Number):
        self._value = value

    def __call__(self, cells: CellSet, t: float) -> np.ndarray:
        # Dimension of the returned array is the size of 1 field. We take the
        # first one [0]
        return np.ones_like(cells.values[..., 0]) * self._value


class BoundaryCondition:
    """A :class:`BoundaryCondition` is a collection of
    :class:`ScalarBC`, one per each :attr:`~.State.fields` of
    the :class:`~.Problem`.

    Attributes
    ----------
    bc
        A :class:`State` instance whose elements per each field are not `float`
        but :class:`ScalarBC` instead

        >>> from josie.state import StateTemplate
        >>> MyState = StateTemplate("u", "v")
        >>> mybc = BoundaryCondition(MyState(Dirichlet(0), Neumann(1)))
    """

    def __init__(self, bc: State):
        self.bc = bc

    def init(self, cells: MeshCellSet, boundary: Boundary):
        """Used to initialize the individual :class:`ScalarBC` per each
        field"""

        boundary_idx = boundary.cells_idx
        boundary_cells = cells[boundary_idx]

        # Apply init BC for each field
        for field in self.bc.fields:
            self.bc[field].init(boundary_cells)

    def __call__(self, cells: MeshCellSet, boundary: Boundary, t: float):
        """
        Parameters
        ----------
        mesh
            A :class:`Mesh` object that contains informations on cells data
            and time instant
        boundary
            A :class:`Boundary` object representing the mesh boundary on which
            the :class:`BoundaryCondition` must be applied
        t
            The time instant to be used to update time dependent
            :class:`ScalarBC`
        """

        ghost_idx = boundary.ghost_cells_idx
        boundary_idx = boundary.cells_idx

        boundary_cells = cells[boundary_idx]
        ghost_cells = cells[ghost_idx]

        # Apply BC for each field
        for field in self.bc.fields:
            cells._values[ghost_idx[0], ghost_idx[1], :, field] = self.bc[field](
                boundary_cells, ghost_cells, field, t
            )


class ScalarBC(abc.ABC):
    """A :class:`ScalarBC` is implemented as a callable that sets the
    equivalent cell value for the ghost cells for the specific field

    """

    def init(self, cells: CellSet):
        """This method is used to initialize the datastructures used to store
        the ghost values and avoid multiple allocations.

        It's useful for constant :class:`ScalarBC` to avoid to recompute the
        same value at each time step: you just store it ones here

        By default it does nothing

        Parameters
        ----------
        cells
            The boundary cells to which this :class:`ScalarBC` is applied

        field
            The field on which this :class:`ScalarBC` is applied (i.e.
            velocity)
        """

        pass

    @abc.abstractmethod
    def __call__(
        self,
        cells: CellSet,
        ghost_cells: CellSet,
        field: int,
        t: float,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        cells:
            A :class:`CellSet` containing the state of the mesh cells

        ghosts_cells
            A :class:`CellSet` containing the ghost cells associated to the
            :class:`Boundary` the :class:`BoundaryCondition` is applied to

        field
            The field to which the :class:`ScalarBC` is applied to
        t
            The time instant to which this :class:`ScalarBC`
            must be evaluated (useful for time-dependent BCs)

        Returns
        -------
        The value of the field on the ghost cell

        """
        raise NotImplementedError


class Dirichlet(ScalarBC):
    r"""A :class:`ScalarBC` that fixes a value of a field on the boundary.

    Assuming we want to impose the value :math:`Q = Q_D` on the (left, as an
    example) boundary, we can assume that the value on the boundary is
    approximated by:

    .. math::

        \frac{Q_{0,j} + Q_\text{ghost}}{2} = Q_D

    That means we can impose the :class:`ScalarBC` assigning the
    value of

    .. math::

        Q_\text{ghost} = 2Q_D - Q_{0,j}

    to the ghost cell.

    Parameters
    ----------
    value
        The value of the field to be imposed on the boundary. It can have
        different types:

        * A :class:`float`. Then it's the scalar constant value for the
        individual field for which we want to apply the BC for. In this case
        the object returned is of type :class:`_ConstantDirichlet` in order to
        optimize since we know it's constant

        * A :class:`BCCallable`. It is called on the cells of the boundary in
        order to provide a non-constant boundary condition

        * A :class:`State` of size :math:`\numberof{fields}`, each element is
        one of the two previous options. Used as a shortcut to impose the same
        :class:`ScalarBC` for all the fields

    constant
        Set this flag to ``True`` to explicitly force the creation of a
        constant boundary condition. A constant BC is optimized to reduce the
        number of calls.

        :class:`Dirichlet` normally is capable to understand automatically if
        you are providing a constant imposed value: if you provide a constant
        scalar :class:`float` or :class:`int` (or a :class:`State` containing a
        scalar value for each field). If you provide a :class:`BCCallable` then
        it cannot automatically infer if your callable is actually only a
        function of space (i.e. it does not change at every time step) or not.
        If you want to optimize the call, you need to explicitly set
        ``constant`` to ``True``.

    Attributes
    ----------
    set_value
        A :class:`SetValueCallable` that can be used to provide a specific
        value the :class:`ScalarBC` needs to impose.
        E.g.
        * In the case of :class:`Dirichlet`, the :attr:`set_value` gives the
        value of the fields to impose on the boundary.

        * In the case of :class:`Neumann` it provides a callable that returns
        the value of the gradient to impose on the boundary
    """

    set_value: BCCallable
    _const_cls = "_ConstantDirichlet"

    def __new__(cls, value: ImposedValue, constant: bool = False):
        # Sorry, this is a bit of black magic to make defining various type of
        # BCs easier on the user side
        if isinstance(value, State):
            # An entire state was provided as value of the Dirichlet BC, so
            # we return a BoundaryCondition with a Dirichlet BC for each
            # field with the right value
            bcs = []

            for field in value.fields:
                bcs.append(cls.__new__(cls, value[field]))

            return BoundaryCondition(np.array(bcs).view(type(value)))

        elif isinstance(value, (int, float)):
            # We transform the value to a callable that returns constant value
            set_value: BCCallable = SetValueCallable(value)
            constant = True

        else:
            # value is already a BCCallable
            set_value = value

        if constant:
            # We return the optimized constant variant
            obj = super().__new__(globals()[cls._const_cls])
            obj.set_value = set_value
            return obj

        # Non-constant variant
        obj = super().__new__(cls)
        obj.set_value = set_value

        return obj

    def __call__(
        self,
        cells: CellSet,
        ghost_cells: CellSet,
        field: int,
        t: float,
    ) -> np.ndarray:
        # FIXME: Ignoring type because of this:
        # https://github.com/python/mypy/issues/708
        return 2 * self.set_value(cells, t) - cells.values[..., field]  # type: ignore # noqa: E501


class _ConstantDirichlet(Dirichlet):
    """An optimized version of :class:`Dirichlet` to be used if the
    :class:`Mesh` is non-dynamic and the value to impose is constant in time.

    It avoids to recompute the value at each time step
    """

    def init(self, cells: CellSet):
        # FIXME: Ignoring type because of this:
        # https://github.com/python/mypy/issues/708
        self._value = self.set_value(cells, 0)  # type: ignore

    def __call__(
        self,
        cells: CellSet,
        ghost_cells: CellSet,
        field: int,
        t: float,
    ) -> np.ndarray:
        return 2 * self._value - cells.values[..., field]


class Neumann(Dirichlet):
    r"""A :class:`ScalarBC` that fixes a value of the gradient of a field on
    the boundary.

    Assuming we want to impose the value of the gradient :math:`\frac{\partial
    Q}{\partial \hat{\mathbf{n}}} = Q_N` of the state on the (left, as an
    example) boundary, we can assume that the value of the gradient on the
    boundary is approximated by:

    .. math::

        \frac{Q_{0,j} - Q_\text{ghost}}{\Delta x} = Q_N

    That means we can impose the :class:`ScalarBC` assigning the
    value of

    .. math::

        Q_\text{ghost} = Q_{0,j} - Q_N

    to the ghost cell (assuming a :math:`\Delta x = 1`)

    Parameters
    ----------
    value
        The value of the field to be imposed on the boundary. It can have
        different types:

        * A :class:`float`. Then it's the scalar constant value for the
        individual field for which we want to apply the BC for

        * A :class:`Callable[[CellSet, t], np.ndarray]`. It is called on the
        cells of the boundary in order to provide a non-constant boundary
        condition

        * A :class:`State` of size :math:`\numberof{fields}`, each element is
        one of the two previous options. Used as a shortcut to impose the same
        :class:`ScalarBC` for all the fields
    """
    _const_cls = "_ConstantNeumann"

    def __call__(
        self,
        cells: CellSet,
        ghost_cells: CellSet,
        field: int,
        t: float,
    ) -> np.ndarray:
        # TODO: Fix for non-zero gradient (need to add dx between the two
        # cells, maybe pre-storing at the beginning if a Neumann BC is given

        # FIXME: Ignoring type because of this:
        # https://github.com/python/mypy/issues/708
        return cells.values[..., field] - self.set_value(cells, t)  # type: ignore # noqa: E501


class _ConstantNeumann(_ConstantDirichlet):
    """An optimized version of :class:`Dirichlet` to be used if the
    :class:`Mesh` is non-dynamic and the value to impose is constant in time.

    It avoids to recompute the value at each time step
    """

    def __call__(
        self,
        cells: CellSet,
        ghost_cells: CellSet,
        field: int,
        t: float,
    ) -> np.ndarray:
        return cells.values[..., field] - self._value


class NeumannDirichlet(ScalarBC):
    """A :class:`ScalarBC` that applies a :class:`Dirichlet` on a part of the
    domain and a :class:`Neumann` on the rest of it.

    Parameters
    ---------
    neumann_value
        The value of the gradient to use for the :class:`Neumann` partition of
        the boundary
    dirichlet_value
        The value of the :class:`State` to use for the :class:`Dirichlet`
        partition of the boundary
    partition_fun
        A callable that takes as input the coordinates of the boundary cells
        and returns the indices of the cells for which the :class:`Dirichlet`
        boundary condition has to be applied
    """

    neumann: Neumann
    dirichlet: Dirichlet
    # partition_fun: Callable[[NeumannDirichlet, np.ndarray], np.ndarray]
    partition_fun: Callable[[np.ndarray], np.ndarray]

    def __new__(
        cls,
        neumann_value: ImposedValue,
        dirichlet_value: ImposedValue,
        partition_fun: Callable[[np.ndarray], np.ndarray],
    ):
        if isinstance(neumann_value, State) and isinstance(dirichlet_value, State):
            # An entire state was provided as value of the NeumannDirichlet BC,
            # so we return a BoundaryCondition with a NeumannDirichlet BC for
            # each field with the right value
            bcs = []

            for field in neumann_value.fields:
                bcs.append(
                    cls.__new__(
                        cls,
                        neumann_value[field],
                        dirichlet_value[field],
                        partition_fun,
                    )
                )

            return BoundaryCondition(np.array(bcs).view(type(neumann_value)))
        else:
            obj = super().__new__(cls)
            obj.neumann = Neumann(neumann_value)
            obj.dirichlet = Dirichlet(dirichlet_value)
            obj.partition_fun = partition_fun  # type: ignore
            return obj

    def init(self, cells: CellSet):
        # Store the partition of cells on which a Diriclet condition is imposed
        self.dirichlet_cells = self.partition_fun(cells.centroids)

        # Init internal state for the Neumann BC first on all the cells
        self.neumann.init(cells)

        # Override only the Dirichlet cells
        self.dirichlet.init(cells[self.dirichlet_cells])

    def __call__(
        self,
        cells: CellSet,
        ghost_cells: CellSet,
        field: int,
        t: float,
    ) -> np.ndarray:
        # First apply Neumann to everything
        ghost_values = self.neumann(cells, ghost_cells, field, t)

        # Apply the Dirichlet BC to the subset of cells
        ghost_values[self.dirichlet_cells] = self.dirichlet(
            cells[self.dirichlet_cells],
            ghost_cells[self.dirichlet_cells],
            field,
            t,
        )

        return ghost_values


class PeriodicSide(NoAliasEnum):
    """A Enum encapsulating the 4 indexing possibilities of a :class:`Periodic`
    :class:`ScalarBC`"""

    LEFT = (-1, slice(None))
    BOTTOM = (slice(None), -1)
    RIGHT = (0, slice(None))
    TOP = (slice(None), 0)


class Periodic(BoundaryCondition):
    r"""A :class:`BoundaryCondition` that connects one side of the domain to
    the other. In general is more straighforward to use the
    :func`make_periodic` function on a couple of :class:`~.BoundaryCurve` that
    needs to be linked periodically

    That means the neighbour cells of the cells on one domain are the cells
    on the other side of the domain. In other words, as an example, given a
    cell on the left boundary, identified by the indices :math:`i,j = (0, j)`,
    i.e. :math:`C_{0,j}`, its west neighbour cell needs to be :math:`C_{N, j}`,
    being :math:`N` the number of cells along the :math:`x`-direction (i.e.
    for increasing index :math:`i`)

    Parameters
    ----------
    side
        The side on which the Periodic BC is configured

    Raises
    ------
    ValueError
        If `side` is not recognized
    """

    def __init__(self, side: PeriodicSide):
        self._side = side

    def init(self, cells: MeshCellSet, boundary: Boundary):
        # Not needed for  Periodic to init anything
        pass

    def __call__(self, cells: MeshCellSet, boundary: Boundary, t: float):
        cells._values[boundary.ghost_cells_idx] = cells.values[self._side.value].copy()


def make_periodic(
    first: BoundaryCurve, second: BoundaryCurve, direction: Direction
) -> Tuple[BoundaryCurve, BoundaryCurve]:
    """This handy function takes as arguments two opposed BoundaryCurve and
    configures them correctly to provide periodic behaviour.

    Parameters
    ----------
    first
        The first :class:`~.BoundaryCurve` to link
    second
        The second :class:`~.BoundaryCurve` to link
    direction
        The direction on which the two :class:`~.BoundaryCurve` are
        periodically connected

    Returns
    -------
    first
        The first :class:`~.BoundaryCurve` whose `bc` attribute is correctly
        configured
    second
        The second :class:`~.BoundaryCurve` whose `bc` attribute is correctly
        configured
    """

    if direction is Direction.X:
        first.bc = Periodic(PeriodicSide.LEFT)
        second.bc = Periodic(PeriodicSide.RIGHT)
    elif direction is Direction.Y:
        first.bc = Periodic(PeriodicSide.BOTTOM)
        second.bc = Periodic(PeriodicSide.TOP)
    else:
        raise ValueError(f"Unknown direction. Expecting a {Direction} object")

    return first, second
