import torch
import transformers
from typing import Optional, Union

from llm4ranking.model.lm.base import LM, LMOuput


class HFLM(LM):
    def __init__(
        self,
        model: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[Union[str, transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None,
        revision: Optional[str] = "main",
        truncation: Optional[bool] = True,
        max_length: Optional[int] = None,
        device_map: Optional[str] = "auto",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = True,
        use_fast_tokenizer: Optional[bool] = False,
        # TODO: PEFT, delta weights and quantization options
        **kwargs,
    ):
        super().__init__()

        # Load model
        if isinstance(model, str):
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                torch_dtype=dtype,
                **kwargs,
            )
        elif isinstance(model, transformers.PreTrainedModel):
            self.model = model
        else:
            raise ValueError(f"Model must be a string or a PreTrainedModel, not {type(model)}")

        # Load tokenizer
        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                    **kwargs,
                )
            elif isinstance(tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast)):
                self.tokenizer = tokenizer
            else:
                raise ValueError(f"Tokenizer must be a string or a PreTrainedTokenizer, not {type(tokenizer)}")
        else:
            assert isinstance(model, str), "Tokenizer must be provided if model is not a string"
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast_tokenizer,
            )

        self._max_length = max_length
        self._truncation = truncation

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return 4096
            return self.tokenizer.model_max_length
        return 4096

    @property
    def max_new_tokens(self):
        return 256

    @property
    def device(self):
        return self.model.device

    def generate(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[str, LMOuput]:
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=self._truncation,
            max_length=self.max_length - max_new_tokens,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )[0, input_ids.shape[-1]:].cpu()
        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)

        if return_num_tokens:
            num_processed_tokens = input_ids.shape[-1]
            num_generated_tokens = outputs.shape[-1]

            return LMOuput(
                text=output_text,
                num_processed_tokens=num_processed_tokens,
                num_generated_tokens=num_generated_tokens,
            )

        return output_text
        

    def loglikelihood(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[float, LMOuput]:
        assert messages[-1]["role"] == "assistant", "Last message must be from the assistant"
        assert len([m for m in messages if m["role"] == "assistant"]) == 1, "Only one assistant message allowed"
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=self._truncation,
            max_length=self.max_length,
            continue_final_message=True,  # this will remove the last eos token
        ).to(self.device)
        labels = self._mask_labels(messages, input_ids.clone())[:, 1:]
        with torch.no_grad():
            logits = self.model(input_ids, **kwargs).logits[:, :-1, :].contiguous().float()
            loglikelihood = -torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).float().item()

        if return_num_tokens:
            num_processed_tokens = input_ids.shape[-1]

            return LMOuput(
                text=messages[-1]["content"],
                loglikelihood=loglikelihood,
                num_processed_tokens=num_processed_tokens,
            )

        return loglikelihood

    def _mask_labels(
        self, 
        messages: list[dict[str, str]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                message_start_idx = self._get_messages_length(messages[:message_idx]) \
                    if message_idx > 0 else 0
                message_end_idx = self._get_messages_length(messages[:message_idx+1])         
                labels[:, message_start_idx:message_end_idx] = -100
                if message_end_idx >= self.tokenizer.model_max_length:
                    break
        return labels

    def _get_messages_length(self, messages: list[dict[str, str]]) -> int:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", truncation=self._truncation,
        ).shape[1]
