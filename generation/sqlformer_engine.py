
class SQLFormerEngine:
    def __init__(self, model, fsm, logits_processor):
        self.model = model
        self.fsm = fsm
        self.logits = logits_processor
    def generate(self, prompt):
        tokens = ['SELECT']
        state = 'SELECT'
        for _ in range(10):
            next_tokens = ['FROM','WHERE','GROUP_BY','END']
            filtered = self.logits.mask(next_tokens, state)
            choice = filtered[0] if filtered else 'END'
            tokens.append(choice)
            state = self.fsm.advance(choice)
            if state == 'END': break
        return " ".join(tokens)
