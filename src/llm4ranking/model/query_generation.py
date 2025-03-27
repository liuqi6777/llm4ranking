from typing import Union

from llm4ranking.lm.base import LMOuput
from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """Please write a question based on this document.

Document: {{ doc }}

Query:"""


class QueryGeneration(BaseRankingModel):
    """Pointwise query generation model for document ranking.
    
    This model evaluates document relevance by attempting to generate the original query
    from the document content. The likelihood of generating the correct query is used
    as a relevance score.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    ranker = "pointwise"
    name = "QueryGeneration"

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        """Create prompt messages for query generation.

        Args:
            query (str): The target query to generate
            doc (str): Document to generate query from

        Returns:
            str: Formatted prompt messages with both system instruction and expected response
        """
        messages = [
            {"role": "user", "content": self.prompt_template.render(doc=doc)},
            {"role": "assistant", "content": query}
        ]
        return messages

    def __call__(self, query: str, doc: str, return_lm_outputs: bool = False) -> Union[float, tuple[float, LMOuput]]:
        """Score a document by measuring how well it generates the target query.

        This method uses the language model's likelihood of generating the original query
        from the document as a measure of relevance.

        Args:
            query (str): The target query that should be generated
            doc (str): Document to evaluate
            return_lm_outputs (bool, optional): Whether to return LM outputs. Defaults to False.

        Returns:
            Union[float, tuple[float, LMOuput]]:
                Returns the log likelihood score of generating the query.
                If return_lm_outputs is True, also returns the LM outputs.
        """
        messages = self.create_messages(query, doc)
        lm_outputs = self.lm.loglikelihood(messages, return_num_tokens=True)
        if return_lm_outputs:
            return lm_outputs.loglikelihood, lm_outputs
        return lm_outputs.loglikelihood
