import os
from typing import Optional, Union, List, Dict

from openai import OpenAI
from llm4ranking.model.lm.base import LM, LMOuput


class OpenAILM(LM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_new_tokens: Optional[int] = 256,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.max_new_tokens = max_new_tokens or 256
        self._max_length = max_length
        self._truncation = truncation

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key must be provided either via `api_key` parameter or `OPENAI_API_KEY` environment variable.")

        self.kwargs = kwargs


    def generate(
        self,
        messages: List[Dict[str, str]],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[str, LMOuput]:
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        api_key = kwargs.pop("api_key", self.api_key)
        API_ENDPOINT = "https://open.xiaojingai.com/v1/"

        client = OpenAI(base_url=API_ENDPOINT,
                        api_key=api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens
        )

        generated_message = response.choices[0].message.content.strip()

        if return_num_tokens:
            num_processed_tokens = sum(len(message["content"].split()) for message in messages)
            num_generated_tokens = len(generated_message.split())

            return LMOuput(
                text=generated_message,
                num_processed_tokens=num_processed_tokens,
                num_generated_tokens=num_generated_tokens,
            )

        return generated_message


    def loglikelihood(
        self,
        messages: List[Dict[str, str]],
        return_num_tokens: Optional[bool] = False,
        **kwargs
    ) -> Union[float, LMOuput]:
        
        raise RuntimeError("OpenAI API does not support loglikelihood calculation")

        # return 0.0
