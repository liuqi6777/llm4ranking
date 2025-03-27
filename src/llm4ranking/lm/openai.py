import os
from typing import Optional, Union, List, Dict

from openai import OpenAI
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

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either via `api_key` or environment variable.")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

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
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=kwargs.pop("max_completion_tokens", self.max_new_tokens),
            **kwargs
        )

        generated_message = response.choices[0].message.content.strip()

        if return_num_tokens:
            num_processed_tokens = response.usage.prompt_tokens
            num_generated_tokens = response.usage.completion_tokens

            return LMOuput(
                text=generated_message,
                num_processed_tokens=num_processed_tokens,
                num_generated_tokens=num_generated_tokens,
            )

        return generated_message

    def loglikelihood(self, **kwargs):
        raise NotImplementedError("OpenAI API does not support loglikelihood calculation")

    def logits(self, **kwargs):
        raise NotImplementedError("OpenAI API does not support logits calculation for now")
