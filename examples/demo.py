
from sqlformer.fsm.sql_fsm import SQLStateMachine
from sqlformer.fsm.transitions import TRANSITIONS
from sqlformer.generation.sqlformer_engine import SQLFormerEngine
from sqlformer.generation.logits_processor import LogitsProcessor
from sqlformer.models.model_loader import DummyModel
fsm = SQLStateMachine(TRANSITIONS)
logits = LogitsProcessor(fsm, None)
model = DummyModel()
engine = SQLFormerEngine(model, fsm, logits)
print(engine.generate("get users"))
