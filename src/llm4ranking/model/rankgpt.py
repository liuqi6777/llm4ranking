import re
from typing import Union

from llm4ranking.lm.base import LMOutput
from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """I will provide you with {{ candidates|length }} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {{ query }}.
{% for content in candidates %}
[{{ loop.index }}] {{content}}
{% endfor %}
Search Query: {{ query }}.
Rank the {{ candidates|length }} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [] > ..., e.g., [4] > [2] > ..., Only respond with the ranking results, do not say anything else or explain.
"""


class RankGPT(BaseRankingModel):
    """RankGPT generates a complete ranking of documents.
    
    This model takes all candidates at once and generates their relative ordering
    based on relevance to the query.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    ranker = "listwise"
    name = "RankGPT"

    def __call__(
        self,
        query: str,
        candidates: list[str],
        return_lm_outputs: bool = False,
        **kwargs
    ) -> Union[list[int], tuple[list[int], LMOutput]]:
        """Generate a ranking for the candidate documents.

        Args:
            query (str): The search query
            candidates (list[str]): List of documents to rank
            return_lm_outputs (bool, optional): Whether to return LM outputs. Defaults to False.
            **kwargs: Additional arguments passed to the LM

        Returns:
            Union[list[int], tuple[list[int], LMOutput]]: 
                Either just the ranking permutation or both the permutation and LM outputs
        """
        messages = self.create_messages(query, candidates)
        lm_outputs = self.lm.generate(messages, **kwargs)
        permutation = self.parse_output(lm_outputs.text, len(candidates))
        if return_lm_outputs:
            return permutation, lm_outputs
        return permutation

    def create_messages(
        self,
        query: str,
        candidates: list[str],
    ) -> str:
        """Create the prompt messages for the LM.

        Args:
            query (str): The search query
            candidates (list[str]): List of documents to rank

        Returns:
            str: Formatted prompt messages
        """
        messages = [
            {"role": "user", "content": self.prompt_template.render(query=query, candidates=candidates)}
        ]
        return messages

    def parse_output(self, output: str, n: int) -> list[int]:
        """Parse the LM output into a ranking permutation.

        Args:
            output (str): Raw output from the LM
            n (int): Number of candidates

        Returns:
            list[int]: Zero-based indices representing the ranking order
        """
        permutation = [int(x) - 1 for x in re.findall(r'\d+', output)]
        permutation = list(dict.fromkeys(permutation))  # remove duplicates
        permutation = [x for x in permutation if x in range(n)] + [x for x in range(n) if x not in permutation]
        return permutation
