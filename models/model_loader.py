import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SQLFormerModel:

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: str = None):
        self.model_name = model_name

        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )

        if self.device.type != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        self.vocab_size = self.model.config.vocab_size

    # get logits for next token prediction
    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor):
        outputs = self.model(input_ids)
        return outputs.logits[:, -1, :]

    def encode(self, text: str):
        return torch.tensor(
            [self.tokenizer.encode(text, add_special_tokens=False)],
            device=self.device
        )

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_to_id(self, token: str):
        return self.tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, idx: int):
        return self.tokenizer.convert_ids_to_tokens(idx)

    @torch.no_grad()
    def next_token_logits(self, prompt: str):
        return self.forward_logits(self.encode(prompt))


def load_sqlformer_model(model_name: str, device: str = None):
    return SQLFormerModel(model_name, device)
