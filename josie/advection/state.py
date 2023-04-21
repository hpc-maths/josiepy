# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from josie.state import SubsetState, State
from josie.fluid.state import ConsState


class AdvectionConsState(SubsetState):
    full_state_fields = State.list_to_enum(["u"])
    fields = State.list_to_enum(["u"])  # type: ignore


class Q(ConsState):
    fields = State.list_to_enum(["u"])  # type: ignore
    cons_state = AdvectionConsState
