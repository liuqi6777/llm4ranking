import re
from jinja2 import Template
from typing import Optional

from llm4ranking.model.lm import load_model

PROMPT_TEMPLATE = ""


class RelevanceGeneration:

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
        input_context = f"Document: {doc}\nQuery:{query}\nIs the document relevant to the query? Give only the result (yes / no), do not give any explanation."
        messages = [{"role": "user", "content": input_context}, {"role": "assistant", "content": "yes"}]
        return messages

    def __call__(self, query: str, doc: str) -> float:
        messages = self.create_messages(query, doc)
        logits = self.lm.loglikelihood(messages)
        return logits
