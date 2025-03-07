from typing import Union

from llm4ranking.lm.base import LMOuput
from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """I will provide you with {{ candidates|length }} documents, each indicated by a numerical identifier. Rank the passages based on their relevance to the search query. All the passages should be listed using identifiers in descending order of relevance. Search Query: {{ query }}.
{% for content in candidates %}
{{ loop.index }}: {{content}}
{% endfor %}
"""


class ListwiseLogit(BaseRankingModel):
    """Ranker that uses the logit of the last input token for listwise ranking."""

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def __call__(
        self,
        query: str,
        candidates: list[str],
        return_lm_outputs: bool = False,
        **kwargs
    ) -> Union[list[int], tuple[list[int], LMOuput]]:
        """Rank the candidaes using logit scores.

        Args:
            query: The search query
            candidates: List of candidate texts to rank
            return_lm_outputs: If True, return both rankings and LM outputs
            **kwargs: Additional arguments for the LM

        Returns:
            Either the ranked indices or a tuple of (ranked indices, LM outputs)
        """
        messages = self.create_messages(query, candidates)
        lm_outputs = self.lm.generate(messages, return_num_tokens=True, **kwargs)
        logits = lm_outputs.logits
        ids = [self.lm.tokenizer.decode([i]) for i in range(len(candidates))]
        logit_for_each_candidate = [logits[int(i)] for i in ids]
        ranking = sorted(range(len(logit_for_each_candidate)), key=lambda i: logit_for_each_candidate[i], reverse=True)
        if return_lm_outputs:
            return ranking, lm_outputs
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
        messages = []
        for candidate in candidates:
            content = self.prompt_template.render(
                query=query,
                candidates=[candidate]  # Process one candidate at a time
            )
            messages.append({"role": "user", "content": content})
        return messages
