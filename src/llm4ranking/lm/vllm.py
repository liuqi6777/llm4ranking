import torch
import vllm
from typing import Optional, Union
from vllm import LLM, SamplingParams

from llm4ranking.lm.base import LM, LMOuput


class VLLM(LM):
    def __init__(
        self,
        model: str,
    ):
        self.model = LLM(model, trust_remote_code=True)

    def generate(
        self,
        messages: list[dict[str, str]],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[str, LMOuput]:
        if return_num_tokens:
            raise NotImplementedError("return_num_tokens is not supported for vLLM engine.")
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        outputs = self.model.chat(
            messages,
            add_generation_prompt=True,
            sampling_params=SamplingParams(
                max_tokens=max_new_tokens,
                **kwargs
            ),
        )
        output_text = outputs[0].outputs[0].text
        return output_text

    def loglikelihood(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[float, LMOuput]:
        raise NotImplementedError("loglikelihood is not supported for vLLM engine.")

    def logits(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("logits is not supported for vLLM engine.")
