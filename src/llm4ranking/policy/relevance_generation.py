import math
from typing import Union

from llm4ranking.lm.base import BatchLMOutput, Capability, LMOutput
from llm4ranking.policy.base import PointwisePolicy, PolicyResult


DEFAULT_PROMPT_TEMPLATE = """Document: {{ doc }}

Query: {{ query }}

Does the document answer the query?
"""


class RelevanceGeneration(PointwisePolicy):
    """Pointwise relevance model that scores individual documents.
    
    This model evaluates each document independently by determining how well
    it answers the query. It uses a yes/no relevance judgment approach and
    the log likelihood of "Yes" response is used as a relevance score.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    name = "RelevanceGeneration"
    supports_batch = True
    required_capabilities = {Capability.LOGITS, Capability.BATCH_LOGITS}

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

    def score(
        self,
        query: str,
        doc: str,
        return_lm_outputs: bool = False,
        **kwargs,
    ) -> Union[float, PolicyResult[float]]:
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

    def parse_output(self, logits: list[float]) -> float:
        yes_prob = math.exp(logits[0])
        no_prob = math.exp(logits[1])
        return yes_prob / (yes_prob + no_prob)

    def parse_batch_outputs(self, lm_outputs: BatchLMOutput) -> list[float]:
        return [self.parse_output(logits) for logits in (lm_outputs.logits or [])]

    def score_many(
        self,
        query: str,
        docs: list[str],
        return_lm_outputs: bool = False,
        **kwargs,
    ) -> Union[list[float], PolicyResult[list[float]]]:
        lm_outputs = self.lm.logits_batch(
            self.create_batch_messages(query, docs),
            token=["yes", "no"],
            **kwargs,
        )
        scores = self.parse_batch_outputs(lm_outputs)
        if return_lm_outputs:
            return self.make_result(scores, lm_outputs)
        return scores


class FineGrainedRelevanceGeneration(RelevanceGeneration):
    """Fine-grained relevance model that scores individual documents.
    
    This model evaluates each document independently by determining how well
    it answers the query. It uses a 5-point scale relevance judgment approach
    and the log likelihood of the selected response is used as a relevance score.
    """

    pass
