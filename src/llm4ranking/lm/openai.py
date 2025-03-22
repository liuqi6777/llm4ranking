import os
from typing import Optional, Union, List, Dict

import litellm
from llm4ranking.lm.base import LM, LMOuput


class OpenAIClient(LM):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()
        self.model = model

        self.api_key = api_key or os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("API key must be provided via `api_key` or environment variables.")

        self.base_url = base_url
        
        self.litellm_params = {
            "model": self.model,
            "api_base": self.base_url,
            "api_key": self.api_key,
            **kwargs
        }

        self._max_length = max_length
        self._truncation = truncation

    @property
    def max_length(self):
        return self._max_length or 4096

    @property
    def max_new_tokens(self):
        return 4096

    def generate(
        self,
        messages: List[Dict[str, str]],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[str, LMOuput]:
        
        response = litellm.completion(
            messages=messages,
            max_tokens=kwargs.pop("max_completion_tokens", self.max_new_tokens),
            **{**self.litellm_params, **kwargs}
        )

        generated_message = response.choices[0].message.content.strip()

        if return_num_tokens:
            num_processed_tokens = response.usage.prompt_tokens if response.usage else 0
            num_generated_tokens = response.usage.completion_tokens if response.usage else 0

            return LMOuput(
                text=generated_message,
                num_processed_tokens=num_processed_tokens,
                num_generated_tokens=num_generated_tokens,
            )

        return generated_message

    def loglikelihood(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Union[float, LMOuput]:
        raise NotImplementedError("LiteLLM API does not support loglikelihood calculation")
    