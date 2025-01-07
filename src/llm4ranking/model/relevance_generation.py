from typing import Union

from llm4ranking.model.lm.base import LMOuput
from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """Document: {{ doc }}

Query: {{ query }}

Does the document answer the query?
"""


class RelevanceGeneration(BaseRankingModel):

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        messages = [
            {"role": "user", "content": self.prompt_template.render(doc=doc, query=query)},
            {"role": "assistant", "content": " Yes"}
        ]
        return messages

    def __call__(self, query: str, doc: str, return_lm_outputs: bool = False) -> Union[float, tuple[float, LMOuput]]:
        messages = self.create_messages(query, doc)
        lm_outputs = self.lm.loglikelihood(messages, return_num_tokens=True)
        if return_lm_outputs:
            return lm_outputs.loglikelihood, lm_outputs
        return lm_outputs.loglikelihood
