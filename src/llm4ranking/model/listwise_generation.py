import re
from jinja2 import Template
from typing import Optional

from llm4ranking.model.lm import load_model


PROMPT_TEMPLATE = """I will provide you with {{ candidates|length }} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {{ query }}.
{% for content in candidates %}
[{{ loop.index }}] {{content}}
{% endfor %}
Search Query: {{ query }}.
Rank the {{ candidates|length }} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [] > ..., e.g., [4] > [2] > ..., Only respond with the ranking results, do not say anything else or explain.
"""


class ListwiseGeneration:

    def __init__(
        self,
        model_type: str,
        model_args: dict,
        prompt_template: Optional[str] = None,
    ):
        self.lm = load_model(model_type, model_args)
        self.template = Template(prompt_template or PROMPT_TEMPLATE)

    def __call__(self, query: str, candidates: list[str], **kwargs) -> dict[str]:
        messages = self.create_messages(query, candidates)
        outputs = self.lm.generate(messages, **kwargs)
        permutation = self.parse_output(outputs, len(candidates))
        return permutation

    def create_messages(
        self,
        query: str,
        candidates: list[str],
    ) -> str:
        messages = [{"role": "user", "content": self.template.render(query=query, candidates=candidates)}]
        return messages

    def parse_output(self, output: str, n: int) -> list[int]:
        permutation = [int(x) - 1 for x in re.findall(r'\d+', output)]
        permutation = list(dict.fromkeys(permutation))  # remove duplicates
        permutation = [x for x in permutation if x in range(n)] + [x for x in range(n) if x not in permutation]
        return permutation
