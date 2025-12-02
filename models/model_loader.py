import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast
)


class SQLFormerModel:
    """
    Unified wrapper around HuggingFace open-source LLMs,
    providing:
        - Tokenization
        - Forward pass (logits only)
        - Vocabulary size & token → ID mapping
        - GPU support
    """

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: str = None):
        self.model_name = model_name

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
        )

        self.model.eval()
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.Tensor):
        """
        Returns raw logits for the last token.

        SQLformer wraps this with a LogitsProcessor that masks
        invalid tokens based on:
            - FSM state
            - Schema constraints
            - Join graph reachability
        """
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1, :]  # last-token logits
        return logits

    def encode(self, text: str):
        """Returns token IDs for input text."""
        return torch.tensor(
            [self.tokenizer.encode(text, add_special_tokens=False)],
            device=self.device
        )

    def decode(self, ids):
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_to_id(self, token: str):
        return self.tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, idx: int):
        return self.tokenizer.convert_ids_to_tokens(idx)

    @torch.no_grad()
    def next_token_logits(self, prompt: str):
        """
        Given a text prompt, encode it and return model logits.
        """
        ids = self.encode(prompt)
        return self.forward_logits(ids)

    def generate_unconstrained(self, prompt: str, max_new_tokens=50):
        """
        Optional helper: normal HF-style generation (no constraints).
        Useful for debugging.
        """
        output = self.model.generate(
            self.encode(prompt),
            max_new_tokens=max_new_tokens
        )
        return self.decode(output[0])

def load_sqlformer_model(model_name: str, device: str = None) -> SQLFormerModel:
    """
    Factory method used by SQLformerEngine.

    Example:
        model = load_sqlformer_model("meta-llama/Meta-Llama-3-8B-Instruct")
    """
    return SQLFormerModel(model_name, device)
