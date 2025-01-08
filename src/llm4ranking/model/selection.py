import re
from typing import Union

from llm4ranking.model.lm.base import LMOuput
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

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def __call__(
        self,
        query: str,
        candidates: list[str],
        num_selection: int,
        return_lm_outputs: bool = False,
        **kwargs
    ) -> Union[list[int], tuple[list[int], LMOuput]]:
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
        messages = [
            {"role": "user", "content": self.prompt_template.render(
                query=query, candidates=candidates, num_selection=num_selection)}
        ]
        return messages

    def parse_output(self, output: str, n: int) -> list[int]:
        idxs = [int(idx) - 1 for idx in re.findall(r"\[(\d+)\]", output)]
        if len(idxs) > n:
            idxs = idxs[:n]
        if len(idxs) < n:
            idxs += [x for x in range(n) if x not in idxs][:n - len(idxs)]
        assert len(idxs) == n
        return idxs
