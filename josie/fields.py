# josiepy
# Copyright © 2021 Ruben Di Battista
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
#
import sys

from typing import List, Optional

from aenum import (
    _is_sunder,
    _is_dunder,
    _is_descriptor,
    _is_private_name,
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

    def __new__(cls, name, bases, clsdict):
        fields = dict(
            [
                (k, Field(k, v))
                for (k, v) in clsdict.items()
                if not (
                    _is_sunder(k)
                    or _is_dunder(k)
                    or _is_private_name(cls, k)
                    or _is_descriptor(v)
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
        """ Functional creation """
        metacls = cls.__class__

        if fields is not None:
            obj = metacls.__new__(metacls, clsname, (cls,), fields)

            # Copied from aenum module
            obj.__module__ = sys._getframe(2).f_globals["__name__"]

            return obj
        else:
            raise TypeError(
                "This class is used like an `Enum`. "
                "Can't be directly instantiated"
            )

    def __getitem__(self, idx):
        return self._field_values[idx]

    def __len__(self):
        return self._len

    def names(self) -> List[str]:
        """ Returns a list of field names """

        return self._field_names


class Fields(metaclass=FieldsMeta):
    pass
