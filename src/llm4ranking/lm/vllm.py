import torch
import vllm
from typing import Optional, Union
from vllm import LLM, SamplingParams

from llm4ranking.lm.base import LM, LMOutput


class VLLM(LM):
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
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[str, LMOutput]:
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        outputs = self.model.generate(
            text,
            use_tqdm=False,
            sampling_params=SamplingParams(
                max_tokens=max_new_tokens,
                **kwargs
            ),
        )
        output_text = outputs[0].outputs[0].text
        return LMOutput(text=output_text)

    def loglikelihood(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[float, LMOutput]:
        raise NotImplementedError("loglikelihood is not supported for vLLM engine.")

    def logits(
        self,
        messages: dict[str, str],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("logits is not supported for vLLM engine.")
