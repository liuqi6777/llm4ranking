from typing import Optional, Union

import numpy as np
import torch
from vllm import LLM, SamplingParams

from llm4ranking.lm.base import BatchLMOutput, Capability, LM, LMOutput


class VLLM(LM):
    supports_batch_generate = True
    supports_batch_loglikelihood = True
    supports_batch_logits = True
    capabilities = {
        Capability.GENERATE,
        Capability.LOGLIKELIHOOD,
        Capability.LOGITS,
        Capability.BATCH_GENERATE,
        Capability.BATCH_LOGLIKELIHOOD,
        Capability.BATCH_LOGITS,
    }

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        trust_remote_code: bool = True,
        tensor_parallel_size: Optional[int] = None,
        enable_prefix_caching: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        chat_template_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        engine_kwargs = dict(kwargs)
        if tokenizer is not None:
            engine_kwargs["tokenizer"] = tokenizer
        engine_kwargs["trust_remote_code"] = trust_remote_code
        engine_kwargs["tensor_parallel_size"] = (
            tensor_parallel_size
            if tensor_parallel_size is not None
            else max(torch.cuda.device_count(), 1)
        )
        engine_kwargs["enable_prefix_caching"] = enable_prefix_caching
        engine_kwargs.setdefault("max_logprobs", -1)

        self.model = LLM(model=model, **engine_kwargs)
        self.tokenizer = self.model.get_tokenizer()
        self._max_length = max_length
        self._truncation = truncation
        self.chat_template_kwargs = {"enable_thinking": False}
        self.chat_template_kwargs.update(chat_template_kwargs or {})

    @property
    def max_length(self) -> int:
        if self._max_length is not None:
            return self._max_length

        engine = getattr(self.model, "llm_engine", None)
        model_config = getattr(engine, "model_config", None)
        model_max_length = getattr(model_config, "max_model_len", None)
        if isinstance(model_max_length, int) and model_max_length > 0:
            return model_max_length

        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 10**9:
            return tokenizer_max_length
        return 4096

    @property
    def max_new_tokens(self) -> int:
        return 256

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> LMOutput:
        batch_output = self.generate_batch([messages], **kwargs)
        return LMOutput(
            text=batch_output.text[0],
            input_tokens=batch_output.input_tokens[0],
            output_tokens=batch_output.output_tokens[0],
        )

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        if not batch_messages:
            return BatchLMOutput(text=[])

        sampling_kwargs, chat_template_kwargs = self._split_request_kwargs(kwargs)
        sampling_params = self._make_sampling_params(
            sampling_kwargs,
            default_max_tokens=self.max_new_tokens,
        )
        prompts = [
            self._tokens_prompt(
                self._truncate_prompt(
                    self._apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    ),
                    reserve_tokens=sampling_params.max_tokens,
                )
            )
            for messages in batch_messages
        ]
        outputs = self.model.generate(
            prompts,
            use_tqdm=False,
            sampling_params=sampling_params,
        )
        return BatchLMOutput(
            text=[output.outputs[0].text for output in outputs],
            input_tokens=[len(prompt["prompt_token_ids"]) for prompt in prompts],
            output_tokens=[len(output.outputs[0].token_ids) for output in outputs],
        )

    def loglikelihood(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> LMOutput:
        batch_output = self.loglikelihood_batch([messages], **kwargs)
        return LMOutput(
            text=batch_output.text[0],
            loglikelihood=batch_output.loglikelihood[0],
            input_tokens=batch_output.input_tokens[0],
            output_tokens=batch_output.output_tokens[0],
        )

    def loglikelihood_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        if not batch_messages:
            return BatchLMOutput(text=[], loglikelihood=[])

        sampling_kwargs, chat_template_kwargs = self._split_request_kwargs(kwargs)
        sampling_kwargs = self._normalize_sampling_kwargs(
            sampling_kwargs,
            default_max_tokens=1,
        )
        sampling_kwargs.pop("logprobs", None)
        sampling_kwargs.pop("logprob_token_ids", None)
        sampling_kwargs.update(
            max_tokens=1,
            temperature=0,
            prompt_logprobs=1,
        )

        prepared = [
            self._prepare_loglikelihood_prompt(messages, chat_template_kwargs)
            for messages in batch_messages
        ]
        outputs = self.model.generate(
            [self._tokens_prompt(token_ids) for token_ids, _ in prepared],
            use_tqdm=False,
            sampling_params=SamplingParams(**sampling_kwargs),
        )

        scores = []
        for output, (token_ids, target_start) in zip(outputs, prepared):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                raise RuntimeError("vLLM did not return requested prompt log probabilities.")
            token_scores = [
                self._extract_logprob(prompt_logprobs[index], token_ids[index])
                for index in range(target_start, len(token_ids))
            ]
            scores.append(sum(token_scores) / len(token_scores))

        return BatchLMOutput(
            text=[messages[-1]["content"] for messages in batch_messages],
            loglikelihood=scores,
            input_tokens=[len(token_ids) for token_ids, _ in prepared],
            output_tokens=[len(token_ids) - target_start for token_ids, target_start in prepared],
        )

    def logits(
        self,
        messages: list[dict[str, str]],
        token: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> LMOutput:
        batch_output = self.logits_batch([messages], token=token, **kwargs)
        return LMOutput(
            logits=batch_output.logits[0],
            input_tokens=batch_output.input_tokens[0],
            output_tokens=batch_output.output_tokens[0],
        )

    def logits_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        token: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> BatchLMOutput:
        if not batch_messages:
            return BatchLMOutput(logits=[])

        sampling_kwargs, chat_template_kwargs = self._split_request_kwargs(kwargs)
        sampling_kwargs = self._normalize_sampling_kwargs(
            sampling_kwargs,
            default_max_tokens=1,
        )
        sampling_kwargs.pop("prompt_logprobs", None)
        sampling_kwargs.pop("logprobs", None)
        sampling_kwargs.pop("logprob_token_ids", None)
        sampling_kwargs.update(
            max_tokens=1,
            temperature=0,
        )
        sampling_params = self._make_logits_sampling_params(sampling_kwargs, token)
        prompts = [
            self._tokens_prompt(
                self._truncate_prompt(
                    self._apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    ),
                    reserve_tokens=1,
                )
            )
            for messages in batch_messages
        ]
        outputs = self.model.generate(
            prompts,
            use_tqdm=False,
            sampling_params=sampling_params,
        )

        batch_scores = []
        for output in outputs:
            logprobs = output.outputs[0].logprobs
            if not logprobs:
                raise RuntimeError("vLLM did not return requested next-token log probabilities.")
            batch_scores.append(self._filter_logprobs(logprobs[0], token))
        return BatchLMOutput(
            logits=batch_scores,
            input_tokens=[len(prompt["prompt_token_ids"]) for prompt in prompts],
            output_tokens=[1] * len(batch_messages),
        )

    def _prepare_loglikelihood_prompt(
        self,
        messages: list[dict[str, str]],
        chat_template_kwargs: dict,
    ) -> tuple[list[int], int]:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("loglikelihood requires a final assistant message to score.")
        if not messages[-1].get("content"):
            raise ValueError("loglikelihood requires non-empty assistant content.")

        prefix_ids = self._apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        token_ids = self._apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
            **chat_template_kwargs,
        )
        if token_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "The tokenizer chat template does not expose a stable assistant boundary."
            )
        if len(token_ids) == len(prefix_ids):
            raise ValueError("The final assistant message produced no tokens to score.")

        truncated_ids = self._truncate_prompt(token_ids)
        removed_tokens = len(token_ids) - len(truncated_ids)
        target_start = len(prefix_ids) - removed_tokens
        if target_start < 1:
            raise ValueError(
                "The assistant response is too long to score within the model context window."
            )
        return truncated_ids, target_start

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> list[int]:
        template_kwargs = dict(self.chat_template_kwargs)
        template_kwargs.update(kwargs)
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            **template_kwargs,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return list(token_ids)

    def _truncate_prompt(
        self,
        token_ids: list[int],
        reserve_tokens: int = 0,
    ) -> list[int]:
        prompt_limit = self.max_length - reserve_tokens
        if prompt_limit < 1:
            raise ValueError(
                f"max_tokens ({reserve_tokens}) must be smaller than max_length ({self.max_length})."
            )
        if len(token_ids) <= prompt_limit:
            return token_ids
        if not self._truncation:
            raise ValueError(
                f"Prompt has {len(token_ids)} tokens but the limit is {prompt_limit}."
            )
        return token_ids[-prompt_limit:]

    def _make_sampling_params(
        self,
        kwargs: dict,
        default_max_tokens: int,
    ) -> SamplingParams:
        return SamplingParams(
            **self._normalize_sampling_kwargs(kwargs, default_max_tokens)
        )

    def _make_logits_sampling_params(
        self,
        sampling_kwargs: dict,
        token: Optional[Union[str, list[str]]],
    ) -> SamplingParams:
        token_ids = self._convert_tokens_to_ids(token)
        if token_ids is None:
            return SamplingParams(**sampling_kwargs, logprobs=-1)

        try:
            return SamplingParams(
                **sampling_kwargs,
                logprobs=len(token_ids),
                logprob_token_ids=token_ids,
            )
        except TypeError as exc:
            if "logprob_token_ids" not in str(exc):
                raise
            return SamplingParams(**sampling_kwargs, logprobs=-1)

    @staticmethod
    def _normalize_sampling_kwargs(
        kwargs: dict,
        default_max_tokens: int,
    ) -> dict:
        sampling_kwargs = dict(kwargs)
        max_new_tokens = sampling_kwargs.pop("max_new_tokens", None)
        max_completion_tokens = sampling_kwargs.pop("max_completion_tokens", None)
        max_tokens = sampling_kwargs.pop("max_tokens", None)
        requested_limits = [
            value
            for value in (max_tokens, max_new_tokens, max_completion_tokens)
            if value is not None
        ]
        if len(requested_limits) > 1:
            raise ValueError(
                "Use only one of max_tokens, max_new_tokens, or max_completion_tokens."
            )
        sampling_kwargs["max_tokens"] = (
            requested_limits[0] if requested_limits else default_max_tokens
        )

        do_sample = sampling_kwargs.pop("do_sample", None)
        if do_sample is False:
            sampling_kwargs.setdefault("temperature", 0)
        return sampling_kwargs

    @staticmethod
    def _split_request_kwargs(kwargs: dict) -> tuple[dict, dict]:
        sampling_kwargs = dict(kwargs)
        chat_template_kwargs = dict(sampling_kwargs.pop("chat_template_kwargs", {}) or {})
        if "enable_thinking" in sampling_kwargs:
            chat_template_kwargs["enable_thinking"] = sampling_kwargs.pop("enable_thinking")
        return sampling_kwargs, chat_template_kwargs

    @staticmethod
    def _tokens_prompt(token_ids: list[int]) -> dict[str, list[int]]:
        return {"prompt_token_ids": token_ids}

    @staticmethod
    def _extract_logprob(logprobs: Optional[dict], token_id: int) -> float:
        if not logprobs or token_id not in logprobs:
            raise RuntimeError(f"vLLM did not return a log probability for token id {token_id}.")
        value = logprobs[token_id]
        return float(getattr(value, "logprob", value))

    def _filter_logprobs(
        self,
        logprobs: dict,
        token: Optional[Union[str, list[str]]],
    ) -> Union[np.ndarray, float, list[float]]:
        if token is None:
            scores = np.full(len(self.tokenizer), -np.inf, dtype=np.float32)
            for token_id, value in logprobs.items():
                if 0 <= token_id < len(scores):
                    scores[token_id] = float(getattr(value, "logprob", value))
            return scores
        if isinstance(token, str):
            return self._extract_logprob(
                logprobs,
                self._convert_tokens_to_ids(token)[0],
            )
        if isinstance(token, list):
            token_ids = self._convert_tokens_to_ids(token)
            return [self._extract_logprob(logprobs, token_id) for token_id in token_ids]
        raise ValueError(f"Token must be a string or a list of strings, not {type(token)}")

    def _convert_tokens_to_ids(
        self,
        token: Optional[Union[str, list[str]]],
    ) -> Optional[list[int]]:
        if token is None:
            return None
        if isinstance(token, str):
            return [self.tokenizer.convert_tokens_to_ids(token)]
        if isinstance(token, list):
            return list(self.tokenizer.convert_tokens_to_ids(token))
        raise ValueError(f"Token must be a string or a list of strings, not {type(token)}")
