import numpy as np
import pytest

from josie.fluid.fields import FluidFields
from josie.fluid.state import SingleFluidState

from josie.fields import Fields
from josie.state import SubsetState


def test_fluid_state():

    # Let's put in different order to check that works indeed
    class AllFields(Fields):
        rho = 0
        rhoU = 1
        U = 2
        rhoV = 3

    class ConsFields(FluidFields):
        rho = 0
        rhoU = 1
        rhoV = 2

    class ConsState(SubsetState):
        full_state_fields = AllFields
        fields = ConsFields

    class MyState(SingleFluidState):
        cons_state = ConsState

    state = np.array([0, 1, 2, 3]).view(MyState)

    cons_state = state.get_conservative().view(ConsState)

    assert cons_state[..., cons_state.fields.rho] == 0
    assert cons_state[..., cons_state.fields.rhoU] == 1
    assert cons_state[..., cons_state.fields.rhoV] == 3

    with pytest.raises(AttributeError):
        cons_state.fields.U
