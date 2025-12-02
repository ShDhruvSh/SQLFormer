from schema.sample_schema import SAMPLE_SCHEMA
from schema.schema_loader import SchemaLoader
from schema.graph_builder import SchemaGraph
from fsm.sql_fsm import SQLStateMachine
from generation.logits_processor import LogitsProcessor
from generation.sqlformer_engine import SQLFormerEngine
from models.model_loader import SQLFormerModel

schema_info = SchemaLoader.from_dict(SAMPLE_SCHEMA)
graph = SchemaGraph(SAMPLE_SCHEMA)
fsm = SQLStateMachine()
logits = LogitsProcessor(fsm, graph, schema_info)

model = SQLFormerModel()  # meta-llama/Meta-Llama-3-8B-Instruct
engine = SQLFormerEngine(model, fsm, logits, schema_info)
print(engine.generate("get users"))