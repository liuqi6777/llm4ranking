from typing import Union

from llm4ranking.lm.base import Capability, LMOutput
from llm4ranking.policy.base import ListwisePolicy, PolicyResult


DEFAULT_PROMPT_TEMPLATE = """I will provide you with {{ candidates|length }} documents, each indicated by a alphabelt identifier. Rank the passages based on their relevance to the search query. All the passages should be listed using identifiers in descending order of relevance. Search Query: {{ query }}.
{% set letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' %}
{% for content in candidates %}
[{{ letters[loop.index0] }}]: {{content}}
{% endfor %}
You should provide the list of identifiers in the order of relevance, for example, "C, A, B, ...", without any additional texts
"""


class First(ListwisePolicy):
    """FIRST reranker that uses the logit of the last input token for listwise ranking."""

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE
    DOCIDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    name = "FIRST"
    required_capabilities = {Capability.LOGITS}

    def rank(
        self,
        query: str,
        candidates: list[str],
        return_lm_outputs: bool = False,
        **kwargs
    ) -> Union[list[int], PolicyResult[list[int]]]:
        """Rank the candidaes using logit scores.

        Args:
            query: The search query
            candidates: List of candidate texts to rank
            return_lm_outputs: If True, return both rankings and LM outputs
            **kwargs: Additional arguments for the LM

        Returns:
            Either the ranked indices or a tuple of (ranked indices, LM outputs)
        """
        if len(candidates) > len(self.DOCIDS):
            raise ValueError(f"Number of candidates ({len(candidates)}) exceeds the maximum number of supported candidates ({len(self.DOCIDS)})")

        messages = self.create_messages(query, candidates)
        lm_outputs = self.lm.logits(messages, **kwargs)
        logits = lm_outputs.logits
        token_ids = [self.lm.tokenizer.encode(i)[0] for i in self.DOCIDS[0:len(candidates)]]
        logit_for_each_candidate = [logits[i] for i in token_ids]
        ranking = sorted(range(len(logit_for_each_candidate)), key=lambda i: logit_for_each_candidate[i], reverse=True)
        if return_lm_outputs:
            return self.make_result(ranking, lm_outputs)
        return ranking

    def create_messages(
        self,
        query: str,
        candidates: list[str],
    ) -> list[dict]:
        """Create messages for each candidate.

        Args:
            query: The search query
            candidates: List of candidate texts to rank

        Returns:
            List of messages, one per candidate
        """
        messages = [
            {"role": "user", "content": self.prompt_template.render(query=query, candidates=candidates)}
        ]
        return messages

    def create_batch_messages(
        self,
        queries: list[str],
        batch_candidates: list[list[str]],
    ) -> list[list[dict[str, str]]]:
        return [self.create_messages(query, candidates) for query, candidates in zip(queries, batch_candidates)]
