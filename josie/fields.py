# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

from typing import List, Optional

from aenum import (
    is_sunder,
    is_dunder,
    is_descriptor,
    is_private_name,
)


class Field(int):
    """A class that acts as a :class:`int` used in :class:`FieldsMeta` to store
    field name and its value"""

    name: str
    value: int

    def __new__(cls, name: str, value: int):
        obj = super().__new__(cls, value)
        obj.name = name
        obj.value = value

        return obj

    def __repr__(self):
        return f"<{self.name}: {self.value}>"


class FieldsMeta(type):
    """This metaclass reproduces in a simpler form the behaviour of
    :class:`Enum`.  It tracks all the defined attributes of a class, it
    precomputes the number of fields and replaces the fields that have no int
    value with the corresponding int
    """

    _field_values: List[Field]
    _field_names: List[str]
    _len: int

    def __new__(cls, name, bases, clsdict):
        fields = dict(
            [
                (k, Field(k, v))
                for (k, v) in clsdict.items()
                if not (
                    is_sunder(k)
                    or is_dunder(k)
                    or is_private_name(cls, k)
                    or is_descriptor(v)
                )
            ]
        )

        # FIXME: Replace `None` indices

        fields_cls = super().__new__(cls, name, bases, fields)

        # Internal data
        fields_cls._field_values = [f for f in fields.values()]
        fields_cls._field_names = [f.name for f in fields.values()]
        fields_cls._len = len(fields)

        return fields_cls

    def __iter__(cls):
        return iter(cls._field_values)

    def __call__(
        cls,
        clsname: Optional[str] = None,
        fields: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """Functional creation"""
        metacls = cls.__class__

        if fields is not None:
            obj = metacls.__new__(metacls, clsname, (cls,), fields)

            # Copied from aenum module
            obj.__module__ = sys._getframe(2).f_globals["__name__"]

            return obj
        else:
            raise TypeError(
                "This class is used like an `Enum`. " "Can't be directly instantiated"
            )

    def __getitem__(self, idx):
        return self._field_values[idx]

    def __len__(self):
        return self._len

    def names(self) -> List[str]:
        """Returns a list of field names"""

        return self._field_names


class Fields(metaclass=FieldsMeta):
    pass
