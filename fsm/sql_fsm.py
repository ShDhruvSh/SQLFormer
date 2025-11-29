
class SQLStateMachine:
    def __init__(self, transitions, start='SELECT'):
        self.transitions = transitions
        self.state = start
    def allowed_next(self):
        return self.transitions.get(self.state, [])
    def advance(self, token):
        if token in self.allowed_next():
            self.state = token
        return self.state
