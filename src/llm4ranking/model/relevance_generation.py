from typing import Union

from llm4ranking.lm.base import LMOuput
from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """Document: {{ doc }}

Query: {{ query }}

Does the document answer the query?
"""


class RelevanceGeneration(BaseRankingModel):
    """Pointwise relevance model that scores individual documents.
    
    This model evaluates each document independently by determining how well
    it answers the query. It uses a yes/no relevance judgment approach and
    the log likelihood of "Yes" response is used as a relevance score.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        """Create prompt messages for relevance assessment.

        Args:
            query (str): The search query
            doc (str): Document to evaluate

        Returns:
            str: Formatted prompt messages
        """
        messages = [
            {"role": "user", "content": self.prompt_template.render(doc=doc, query=query)},
            {"role": "assistant", "content": " Yes"}
        ]
        return messages

    def __call__(self, query: str, doc: str, return_lm_outputs: bool = False) -> Union[float, tuple[float, LMOuput]]:
        """Score a document based on its relevance to the query.

        Args:
            query (str): The search query
            doc (str): Document to evaluate
            return_lm_outputs (bool, optional): Whether to return LM outputs. Defaults to False.

        Returns:
            Union[float, tuple[float, LMOuput]]:
                Returns a relevance score (log likelihood of "Yes" response).
                If return_lm_outputs is True, also returns the LM outputs.
        """
        messages = self.create_messages(query, doc)
        lm_outputs = self.lm.loglikelihood(messages, return_num_tokens=True)
        if return_lm_outputs:
            return lm_outputs.loglikelihood, lm_outputs
        return lm_outputs.loglikelihood
