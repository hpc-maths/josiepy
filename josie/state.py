# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from typing import Collection, Union, Type, TYPE_CHECKING

from .fields import Fields


if TYPE_CHECKING:
    from josie.mesh import Mesh


def unpickle_state(d, array):
    Q = StateTemplate(*d.keys())
    state = array.view(Q)

    return state


class State(np.ndarray):
    """:class:`State` is a subclass of :class:`numpy.ndarray`. It behaves like
    a normal :class:`numpy.ndarray` except it has additional init methods to
    ease the usage

    A :class:`State` can be initialized using a :class:`StateTemplate`,

    >>> Q = StateTemplate("rho", "rhoU", "rhoV")
    >>> state = np.array([0, 1, 2]).view(Q)
    >>> assert state[state.fields.rho] == 0
    >>> assert state[state.fields.rhoU] == 1
    >>> assert state[state.fields.rhoV] == 2

    or directly providing key-value arguments

    >>> state = State(rho=0, rhoU=1, rhoV=2)
    >>> assert state[state.fields.rho] == 0
    >>> assert state[state.fields.rhoU] == 1
    >>> assert state[state.fields.rhoV] == 2

    A :class:`State` can also store non-numeric elements. Check
    :class:`~.bc.BoundaryCondition` for an actual situation in which callables
    are stored instead of numeric elements.

    A :class:`State can be manipulated as a normal :class:`numpy.ndarray`:

    >>> e1 = State([1, 0, 0])
    >>> e2 = State([0, 1, 0])
    >>> e3 = State([0, 0, 1])
    >>> assert np.array_equal(np.cross(e1, e2), e3)

    A :class:`State` can be multidimensional. The last dimension must be the
    number of states defined in the :class:`StateTemplate` call. In this case
    you can get all the values of the state for a specific variable:

    >>> state = np.random.random((10, 10, 3)).view(Q)
    >>> assert np.array_equal(state[..., state.fields.rho], state[..., 0])
    >>> assert np.array_equal(state[..., state.fields.rhoU],state[..., 1])
    """

    fields: Type[Fields]
    _FIELDS_ENUM_NAME = "FieldsEnum"

    def __new__(cls, *args, **kwargs):
        if args and kwargs:
            raise TypeError(
                "A State can be defined using positional arguments OR "
                "keyword arguments, not both"
            )

        if kwargs:
            cls.fields = cls.list_to_enum(kwargs.keys())
            args = tuple(kwargs.values())

        if isinstance(args[0], (int, float)):
            dtype: Union[Type[float], Type[object]] = float
        else:
            dtype = object

        return np.asarray(list(args), dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        # Normal __new__ construction
        if obj is None:
            return

        self._getitem = False

        # Slice handled by superclass
        if isinstance(obj, State) and obj._getitem:
            return

    def __reduce__(self):
        # Let's override the pickling behaviour replacing the enum with a dict
        enum_dict = {f.name: f.value for f in self.fields}
        return (unpickle_state, (enum_dict, self.__array__()))

    @classmethod
    def list_to_enum(cls, fields: Collection[str]) -> Type[Fields]:
        """Convert a list of textual fields to the class:`IntEnum` that needs
        to be stored in this class :attr:`fields`"""

        return Fields(  # type: ignore
            cls._FIELDS_ENUM_NAME, dict(zip(fields, range(len(fields))))
        )  # type: ignore

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> State:
        """Initialize an empty class:`State` object of the right dimensions
        for the given class:`Mesh`"""

        # TODO: Add num_dofs into the size to allow for multiple dofs in a
        # single cell
        nx = mesh.num_cells_x
        ny = mesh.num_cells_y
        num_dofs = mesh.cell_type.num_dofs
        state_size = len(cls.fields)

        return np.empty((nx + 2, ny + 2, num_dofs, state_size)).view(  # type: ignore[arg-type]
            cls
        )


def StateTemplate(*fields: str) -> Type[State]:
    r"""A factory for a :class:`State`.

    It allows you to create at will a :class:`State` class for which you can
    access its variables (e.g. the velocity :math:`\mathbf{U}`) using the
    attribute :attr:`fields`, that is an :class:`IntEnum` (and not only by
    index).

    Parameters
    ----------
    fields
        A list of (scalar) fields composing the state


    A scalar :class:`State` as for the advection equation

    >>> Q = StateTemplate("u")

    Than you can concretize the state with a value

    >>> zero = Q(0)

    You can also create higher dimensional states, for examples the state
    of the 2D Euler compressible equations

    >>> Q = StateTemplate("rho", "rhoU", "rhoV", "E")
    >>> zero = Q(0, 0, 0, 0)
    >>> assert zero[Q.fields.rho] == 0
    """
    # Dynamically create a class of type "State" (actually a subclass)
    # with the right :attr:`fields`
    state_fields: Type[Fields] = State.list_to_enum(fields)
    state_cls = type("DerivedState", (State,), {"fields": state_fields})  # type: ignore # noqa: E501

    return state_cls


class SubsetState(State):
    """This class stores a subset of :class`State`, filtered by field name. It also
    stores a mapping of the indices in :attr:`State.fields` corresponding
    to the :attr:`SubsetState.fields` in :attr:`self._subset_fields_map`

    Attributes
    ----------
    full_state_fields
        The fields of the full :class:`State`
    """

    if TYPE_CHECKING:
        _subset_fields_map: Union[np.typing.ArrayLike, dict]
        full_state_fields: Type[Fields]

    def __init_subclass__(cls, /, abstract=False, **kwargs):
        super.__init_subclass__(**kwargs)

        if not (abstract):
            # FIXME: Using an array to subset the full state creates a copy of
            # the array, while using a slice generates a view
            cls._subset_fields_map = np.array(
                [
                    field
                    for field in cls.full_state_fields
                    if field.name in cls.fields.names()
                ]
            )
