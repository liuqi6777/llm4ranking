from jinja2 import Template
from typing import Optional

from llm4ranking.model.lm import load_model


PROMPT_TEMPLATE = """Please write a question based on this document.

Document: {{ doc }}

Query:"""


class QueryGeneration:

    def __init__(
        self,
        model_type: str,
        model_args: dict,
        prompt_template: Optional[str] = None,
    ):
        self.lm = load_model(model_type, model_args)
        self.template = Template(prompt_template or PROMPT_TEMPLATE)

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        messages = [{"role": "user", "content": self.template.render(doc=doc)}, {"role": "assistant", "content": query}]
        return messages

    def __call__(self, query: str, doc: str) -> float:
        messages = self.create_messages(query, doc)
        score = self.lm.loglikelihood(messages)
        return score
