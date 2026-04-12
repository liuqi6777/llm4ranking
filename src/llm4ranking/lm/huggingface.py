import numpy as np
import torch
import transformers
from typing import Optional, Union
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence

from llm4ranking.lm.base import BatchLMOutput, LM, LMOutput


class HFLM(LM):
    supports_batch_generate = True
    supports_batch_loglikelihood = True
    supports_batch_logits = True

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
        # PEFT options
        peft: Optional[str] = None,
        # Delta weights options
        delta: Optional[str] = None,
        # Quantization options
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        quantization_config: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize the HFLM model.

        Args:
            model: The model name or path, or a PreTrainedModel instance
            tokenizer: The tokenizer name or path, or a PreTrainedTokenizer instance
            revision: The revision of the model to use
            truncation: Whether to truncate the input
            max_length: The maximum length of the input
            device_map: The device map to use
            dtype: The data type to use
            trust_remote_code: Whether to trust the remote code
            use_fast_tokenizer: Whether to use a fast tokenizer
            peft: The path to the PEFT model to use
            delta: The path to the delta weights to use
            quantization_config: The quantization configuration to use
            load_in_8bit: Whether to load the model in 8-bit
            load_in_4bit: Whether to load the model in 4-bit
            **kwargs: Additional keyword arguments
        """
        super().__init__()

        # Load model
        if isinstance(model, str):
            # Prepare quantization config if specified
            if quantization_config or load_in_8bit or load_in_4bit:
                if load_in_8bit and load_in_4bit:
                    raise ValueError("Cannot load model in both 8-bit and 4-bit")
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    **(quantization_config or {})
                )

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                torch_dtype=dtype,
                quantization_config=quantization_config,
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
            )

        if peft and delta:
            raise ValueError(
                "Cannot use both 'peft' and 'delta' options at the same time."
            )

        if peft:
            if self.model.config.vocab_size != len(self.tokenizer):
                # Resize model for LoRAs with added tokens.
                self.model.resize_token_embeddings(len(self.tokenizer))
            self.model = PeftModel.from_pretrained(
                self.model, peft, revision=revision
            )
        elif delta:
            model_delta = transformers.AutoModelForCausalLM.from_pretrained(
                delta,
                revision=revision,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
            for name, param in self.model.state_dict().items():
                try:
                    param.data += model_delta.state_dict()[name]
                except KeyError:
                    raise KeyError(f"Delta model is missing weights for layer: {name}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to add delta weights to layer {name}. Error: {e}"
                    )

        self._max_length = max_length
        self._truncation = truncation

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for batching.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        messages: list[dict[str, str]],
        **kwargs
    ) -> LMOutput:
        batch_outputs = self.generate_batch([messages], **kwargs)
        return LMOutput(text=batch_outputs.text[0])

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        if not batch_messages:
            return BatchLMOutput(text=[])

        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        input_ids, attention_mask = self._prepare_generation_inputs(batch_messages, max_new_tokens=max_new_tokens)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **self._prepare_generation_kwargs(kwargs),
            )[:, input_ids.shape[-1]:].cpu()

        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return BatchLMOutput(text=output_texts)


    def loglikelihood(
        self,
        messages: dict[str, str],
        **kwargs
    ) -> LMOutput:
        """Get the loglikelihood of the model.

        Args:
            messages: The messages to get the loglikelihood from
            return_num_tokens: Whether to return the number of tokens
            **kwargs: Additional keyword arguments

        Returns:
            Either the loglikelihood or a LMOutput object containing the loglikelihood and the number of tokens
        """
        batch_outputs = self.loglikelihood_batch([messages], **kwargs)
        return LMOutput(
            text=batch_outputs.text[0],
            loglikelihood=batch_outputs.loglikelihood[0],
        )

    def loglikelihood_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        if not batch_messages:
            return BatchLMOutput(text=[], loglikelihood=[])

        input_ids_list = [self._prepare_loglikelihood_input_ids(messages).squeeze(0) for messages in batch_messages]
        labels_list = [
            self._mask_labels(messages, input_ids.unsqueeze(0).clone())[:, 1:].squeeze(0)
            for messages, input_ids in zip(batch_messages, input_ids_list)
        ]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100).to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        loglikelihoods = self._compute_loglikelihoods(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs,
        )
        return BatchLMOutput(
            text=[messages[-1]["content"] for messages in batch_messages],
            loglikelihood=loglikelihoods,
        )

    def logits(
        self,
        messages: dict[str, str],
        token: Optional[str] = None,
        **kwargs
    ) -> LMOutput:
        """Get the logits of the model.

        Args:
            messages: The messages to get the logits from
            return_num_tokens: Whether to return the number of tokens
            **kwargs: Additional keyword arguments

        Returns:
            Either the logits of the last token of the input messages or a LMOutput object containing the logits and the number of tokens
        """
        batch_outputs = self.logits_batch([messages], token=token, **kwargs)
        return LMOutput(logits=batch_outputs.logits[0])

    def logits_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        token: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> BatchLMOutput:
        if not batch_messages:
            return BatchLMOutput(logits=[])

        input_ids_list = [self._prepare_logits_input_ids(messages).squeeze(0) for messages in batch_messages]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            logits = outputs.logits.detach().float().cpu()

        last_indices = attention_mask.sum(dim=-1).cpu() - 1
        batch_logits = []
        for batch_idx, last_idx in enumerate(last_indices.tolist()):
            one_logits = logits[batch_idx, last_idx, :].numpy()
            batch_logits.append(self._filter_logits(one_logits, token))
        return BatchLMOutput(logits=batch_logits)

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

    def _prepare_generation_kwargs(self, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if kwargs.get("do_sample") is False:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_k = None
            self.model.generation_config.top_p = None
        return kwargs

    def _prepare_generation_inputs(
        self,
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids_list = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=self._truncation,
                max_length=self.max_length - max_new_tokens,
                enable_thinking=False,
            ).squeeze(0)
            for messages in batch_messages
        ]
        max_len = max(input_ids.size(0) for input_ids in input_ids_list)
        padded_input_ids = []
        padded_attention_mask = []
        for input_ids in input_ids_list:
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                padding = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                input_ids = torch.cat([padding, input_ids], dim=0)
                attention_mask = torch.cat(
                    [
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(input_ids.size(0) - pad_len, dtype=torch.long),
                    ],
                    dim=0,
                )
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
        return (
            torch.stack(padded_input_ids, dim=0).to(self.device),
            torch.stack(padded_attention_mask, dim=0).to(self.device),
        )

    def _prepare_loglikelihood_input_ids(self, messages: list[dict[str, str]]) -> torch.Tensor:
        return self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=self._truncation,
            max_length=self.max_length,
            continue_final_message=True,
            enable_thinking=False,
        )

    def _prepare_logits_input_ids(self, messages: list[dict[str, str]]) -> torch.Tensor:
        return self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=self._truncation,
            max_length=self.max_length,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _compute_loglikelihoods(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> list[float]:
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask, **kwargs).logits[:, :-1, :].contiguous().float()
            losses = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                labels,
                ignore_index=-100,
                reduction="none",
            )
        valid_mask = labels.ne(-100)
        token_counts = valid_mask.sum(dim=-1).clamp(min=1)
        loss_sums = (losses * valid_mask).sum(dim=-1)
        return (-(loss_sums / token_counts)).cpu().tolist()

    def _filter_logits(
        self,
        logits: np.ndarray,
        token: Optional[Union[str, list[str]]],
    ) -> Union[np.ndarray, float, list[float]]:
        if token is None:
            return logits
        if isinstance(token, str):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            return float(logits[token_id].item())
        if isinstance(token, list):
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            return [float(logits[token_id].item()) for token_id in token_ids]
        raise ValueError(f"Token must be a string or a list of strings, not {type(token)}")
