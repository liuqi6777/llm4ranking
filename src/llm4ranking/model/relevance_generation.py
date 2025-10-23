from typing import Union

from llm4ranking.lm.base import LMOutput
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

    ranker = "pointwise"
    name = "RelevanceGeneration"

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
            # {"role": "assistant", "content": "yes"}
        ]
        return messages

    def __call__(self, query: str, doc: str, return_lm_outputs: bool = False) -> Union[float, tuple[float, LMOutput]]:
        """Score a document based on its relevance to the query.

        Args:
            query (str): The search query
            doc (str): Document to evaluate
            return_lm_outputs (bool, optional): Whether to return LM outputs. Defaults to False.

        Returns:
            Union[float, tuple[float, LMOutput]]:
                Returns a relevance score (log likelihood of "Yes" response).
                If return_lm_outputs is True, also returns the LM outputs.
        """
        messages = self.create_messages(query, doc)
        lm_outputs = self.lm.logits(messages, token="yes", return_num_tokens=True)
        if return_lm_outputs:
            return lm_outputs.logits, lm_outputs
        return lm_outputs.logits


class FineGrainedRelevanceGeneration(RelevanceGeneration):
    """Fine-grained relevance model that scores individual documents.
    
    This model evaluates each document independently by determining how well
    it answers the query. It uses a 5-point scale relevance judgment approach
    and the log likelihood of the selected response is used as a relevance score.
    """

    pass
