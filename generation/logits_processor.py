
class LogitsProcessor:
    def __init__(self, fsm, constraints):
        self.fsm = fsm
        self.constraints = constraints
    def mask(self, tokens, state):
        return [t for t in tokens if t.startswith(state[0].lower()) or len(t)<5]
