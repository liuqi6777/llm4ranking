import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm4ranking.model.lm.base import LM


class HFLM(LM):
    def __init__(
        self,
        model_name_or_path: str = None,
        model = None,
        tokenizer = None,
    ):
        super().__init__()
        self.model_name = model_name_or_path
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            assert model_name_or_path is not None, "model_name_or_path must be provided."
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                padding_side="right"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def generate(
        self,
        messages: dict[str, str],
        **kwargs
    ) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )[0, input_ids.shape[-1]:].cpu()
        return self.tokenizer.decode(outputs, skip_special_tokens=True)

    def loglikelihood(
        self,
        messages: dict[str, str],
        **kwargs
    ) -> float:
        pass

    def _mask_labels(
        self, 
        messages: list[dict[str, str]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                message_start_idx = self._get_messages_length(messages[:message_idx]) if message_idx > 0 else 0
                message_end_idx = self._get_messages_length(messages[:message_idx+1])         
                labels[:, message_start_idx:message_end_idx] = -100
                if message_end_idx >= self.tokenizer.model_max_length:
                    break
        return labels

    def _get_messages_length(self, messages: list[dict[str, str]]) -> int:
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=True).shape[1]

