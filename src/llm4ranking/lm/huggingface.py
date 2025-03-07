import torch
import transformers
from typing import Optional, Union

from llm4ranking.lm.base import LM, LMOuput


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
        # PEFT options
        peft_config: Optional[dict] = None,
        # Delta weights options
        delta_weights_path: Optional[str] = None,
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
            peft_config: The PEFT configuration to use
            delta_weights_path: The path to the delta weights to use
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

            # Apply PEFT if config provided
            if peft_config:
                from peft import get_peft_model, prepare_model_for_kbit_training
                if quantization_config:
                    self.model = prepare_model_for_kbit_training(self.model)
                self.model = get_peft_model(self.model, peft_config)

            # Load delta weights if provided
            if delta_weights_path:
                delta_state_dict = torch.load(delta_weights_path, map_location="cpu")
                self.model.load_state_dict(delta_state_dict, strict=False)

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
        """Generate text from the model.

        Args:
            messages: The messages to generate from
            return_num_tokens: Whether to return the number of tokens
            **kwargs: Additional keyword arguments

        Returns:
            Either the generated text or a LMOuput object containing the text and the number of tokens
        """
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

    def logits(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        """Get the logits of the model.

        Args:
            messages: The messages to get the logits from
            return_num_tokens: Whether to return the number of tokens
            **kwargs: Additional keyword arguments

        Returns:
            Either the logits of the last token of the input messages or a LMOuput object containing the logits and the number of tokens
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            truncation=self._truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            logits = outputs.logits[0, -1, :]
        if return_num_tokens:
            num_processed_tokens = input_ids.shape[-1]
            return LMOuput(
                text=messages[-1]["content"],
                logits=logits,
                num_processed_tokens=num_processed_tokens,
            )
        return logits

    def loglikelihood(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[float, LMOuput]:
        """Get the loglikelihood of the model.

        Args:
            messages: The messages to get the loglikelihood from
            return_num_tokens: Whether to return the number of tokens
            **kwargs: Additional keyword arguments

        Returns:
            Either the loglikelihood or a LMOuput object containing the loglikelihood and the number of tokens
        """
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
