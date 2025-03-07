import re
from typing import Union

from llm4ranking.lm.base import LMOuput
from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """I will provide you with the given query and {{ candidates|length }} documents, each indicated by a numerical identifier [].

Consider the content of all the documents comprehensively and select the {{ num_selection }} documents that are most relevant to the given query: {{ query }}.

{% for content in candidates %}
[{{ loop.index }}] {{content}}
{% endfor %}

Search Query: {{ query }}.

Output the {{ num_selection }} unique documents that are most relevant to the Query. The output format should be: "[3], ..., [1]" (with {{ num_selection }} numbers). Do not say anything else or explain.
"""


class Selection(BaseRankingModel):
    """Tournament selection model that picks top documents from a set.
    
    This model takes a set of documents and directly selects a specified number
    of most relevant documents, rather than producing a full ranking.
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def __call__(
        self,
        query: str,
        candidates: list[str],
        num_selection: int,
        return_lm_outputs: bool = False,
        **kwargs
    ) -> Union[list[int], tuple[list[int], LMOuput]]:
        """Select the most relevant documents from a candidate set.

        Args:
            query (str): The search query
            candidates (list[str]): List of documents to select from
            num_selection (int): Number of documents to select
            return_lm_outputs (bool, optional): Whether to return LM outputs. Defaults to False.
            **kwargs: Additional arguments passed to the LM

        Returns:
            Union[list[int], tuple[list[int], LMOuput]]:
                Returns indices of selected documents.
                If return_lm_outputs is True, also returns the LM outputs.
        """
        messages = self.create_messages(query, candidates, num_selection)
        lm_outputs = self.lm.generate(messages, return_num_tokens=True, **kwargs)
        seleted_idx = self.parse_output(lm_outputs.text, num_selection)
        if return_lm_outputs:
            return seleted_idx, lm_outputs
        return seleted_idx

    def create_messages(
        self,
        query: str,
        candidates: list[str],
        num_selection: int,
    ) -> str:
        """Create prompt messages for document selection.

        Args:
            query (str): The search query
            candidates (list[str]): List of documents to select from
            num_selection (int): Number of documents to select

        Returns:
            str: Formatted prompt messages
        """
        messages = [
            {"role": "user", "content": self.prompt_template.render(
                query=query, candidates=candidates, num_selection=num_selection)}
        ]
        return messages

    def parse_output(self, output: str, n: int) -> list[int]:
        """Parse the LM output into selected document indices.

        Args:
            output (str): Raw output from the LM
            n (int): Number of documents to select

        Returns:
            list[int]: Indices of selected documents
        """
        idxs = [int(idx) - 1 for idx in re.findall(r"\[(\d+)\]", output)]
        if len(idxs) > n:
            idxs = idxs[:n]
        if len(idxs) < n:
            idxs += [x for x in range(n) if x not in idxs][:n - len(idxs)]
        assert len(idxs) == n
        return idxs
