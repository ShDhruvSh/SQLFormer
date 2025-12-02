class SQLStateMachine:
    TRANSITIONS = {
        "START": ["SELECT"],
        "SELECT": ["FROM"],
        "FROM": ["WHERE", "GROUP_BY", "ORDER_BY", "END"],
        "WHERE": ["GROUP_BY", "ORDER_BY", "END"],
        "GROUP_BY": ["ORDER_BY", "END"],
        "ORDER_BY": ["END"],
        "END": []
    }

    def __init__(self, start='START'):
        self.state = start
        self.completed = False

    def allowed_next(self):
        return self.TRANSITIONS[self.state]

    def advance(self, token):
        if token in self.allowed_next():
            self.state = token
        if self.state == "END":
            self.completed = True
        return self.state