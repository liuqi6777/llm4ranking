import os
from typing import Optional, Union, List, Dict

from openai import OpenAI
from llm4ranking.lm.base import BatchLMOutput, Capability, LM, LMOutput


class OpenAIClient(LM):
    supports_batch_generate = True
    capabilities = {Capability.GENERATE, Capability.BATCH_GENERATE}

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
            api_key=self.api_key,
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
        **kwargs
    ) -> LMOutput:
        batch_output = self.generate_batch([messages], **kwargs)
        return LMOutput(text=batch_output.text[0])

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        texts = []
        for messages in batch_messages:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=kwargs.get("max_completion_tokens", self.max_new_tokens),
                **{key: value for key, value in kwargs.items() if key != "max_completion_tokens"}
            )
            texts.append(response.choices[0].message.content.strip())
        return BatchLMOutput(text=texts)

    def loglikelihood(self, **kwargs):
        raise NotImplementedError("OpenAI API does not support loglikelihood calculation")

    def logits(self, **kwargs):
        raise NotImplementedError("OpenAI API does not support logits calculation for now")
