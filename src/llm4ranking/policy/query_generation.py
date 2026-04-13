from typing import Union

from llm4ranking.lm.base import BatchLMOutput, Capability, LMOutput
from llm4ranking.policy.base import PointwisePolicy, PolicyResult


DEFAULT_PROMPT_TEMPLATE = """Please write a question based on this document.

Document: {{ doc }}

Query:"""


class QueryGeneration(PointwisePolicy):
    """Pointwise query generation model for document ranking.
    
    This model evaluates document relevance by attempting to generate the original query
    from the document content. The likelihood of generating the correct query is used
    as a relevance score.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    name = "QueryGeneration"
    supports_batch = True
    required_capabilities = {Capability.LOGLIKELIHOOD, Capability.BATCH_LOGLIKELIHOOD}

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

    def score(
        self,
        query: str,
        doc: str,
        return_lm_outputs: bool = False,
        **kwargs,
    ) -> Union[float, PolicyResult[float]]:
        """Score a document by measuring how well it generates the target query.

        This method uses the language model's likelihood of generating the original query
        from the document as a measure of relevance.

        Args:
            query (str): The target query that should be generated
            doc (str): Document to evaluate
            return_lm_outputs (bool, optional): Whether to return LM outputs. Defaults to False.

        Returns:
            Union[float, tuple[float, LMOutput]]:
                Returns the log likelihood score of generating the query.
                If return_lm_outputs is True, also returns the LM outputs.
        """
        result = self.score_many(
            query,
            [doc],
            return_lm_outputs=True,
            **kwargs,
        )
        if return_lm_outputs:
            return self.make_result(result.value[0], self._unwrap_batch_output(result.lm_outputs))
        return result.value[0]

    def create_batch_messages(
        self,
        query: str,
        docs: list[str],
    ) -> list[list[dict[str, str]]]:
        return [self.create_messages(query, doc) for doc in docs]

    def parse_batch_outputs(self, lm_outputs: BatchLMOutput) -> list[float]:
        return lm_outputs.loglikelihood or []

    def score_many(
        self,
        query: str,
        docs: list[str],
        return_lm_outputs: bool = False,
        **kwargs,
    ) -> Union[list[float], PolicyResult[list[float]]]:
        lm_outputs = self.lm.loglikelihood_batch(self.create_batch_messages(query, docs), **kwargs)
        scores = self.parse_batch_outputs(lm_outputs)
        if return_lm_outputs:
            return self.make_result(scores, lm_outputs)
        return scores
