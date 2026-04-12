import torch
import vllm
from typing import Optional, Union
from vllm import LLM, SamplingParams

from llm4ranking.lm.base import BatchLMOutput, LM, LMOutput


class VLLM(LM):
    supports_batch_generate = True

    def __init__(
        self,
        model: str,
        **kwargs,
    ):
        self.model = LLM(
            model=model,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_prefix_caching=True,
            **kwargs
        )
        self.tokenizer = self.model.get_tokenizer()

    def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs
    ) -> Union[str, LMOutput]:
        batch_output = self.generate_batch([messages], **kwargs)
        return LMOutput(text=batch_output.text[0])

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        if not batch_messages:
            return BatchLMOutput(text=[])

        max_new_tokens = kwargs.pop("max_new_tokens", None)
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for messages in batch_messages
        ]
        outputs = self.model.generate(
            prompts,
            use_tqdm=False,
            sampling_params=SamplingParams(
                max_tokens=max_new_tokens,
                **kwargs
            ),
        )
        return BatchLMOutput(text=[output.outputs[0].text for output in outputs])

    def loglikelihood(
        self,
        messages: dict[str, str],
        **kwargs
    ) -> Union[float, LMOutput]:
        raise NotImplementedError("loglikelihood is not supported for vLLM engine.")

    def logits(
        self,
        messages: dict[str, str],
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("logits is not supported for vLLM engine.")
