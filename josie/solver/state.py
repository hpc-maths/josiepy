class State:
    def __init__(self, **state_vars):
        for name, val in state_vars.items():
            setattr(self, name, val)
