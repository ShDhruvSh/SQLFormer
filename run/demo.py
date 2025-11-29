
from fsm.sql_fsm import SQLStateMachine
from fsm.transitions import TRANSITIONS
from generation.sqlformer_engine import SQLFormerEngine
from generation.logits_processor import LogitsProcessor
from models.model_loader import DummyModel
fsm = SQLStateMachine(TRANSITIONS)
logits = LogitsProcessor(fsm, None)
model = DummyModel()
engine = SQLFormerEngine(model, fsm, logits)
print(engine.generate("get users"))
