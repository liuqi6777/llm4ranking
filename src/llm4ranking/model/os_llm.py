import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class OSLlm:
    def __init__(
        self,
        model_name_or_path: str,
    ):
        super().__init__()
        self.model_name = model_name_or_path
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
        self.eval()

    def eval(self):
        self.model.eval()

    @torch.no_grad()
    def get_response(
        self,
        messages: dict[str, str],
        max_length: int = 4096,
        max_new_tokens: int = 128,
        **kwargs
    ) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            **kwargs
        )[0, input_ids.shape[-1]:].cpu()
        return self.tokenizer.decode(outputs, skip_special_tokens=True)

    def create_messages(self, **kwargs) -> str:
        raise NotImplementedError

