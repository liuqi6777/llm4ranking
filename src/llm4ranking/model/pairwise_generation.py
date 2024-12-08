import re
from jinja2 import Template
from typing import Optional

from llm4ranking.model.lm import load_model

PROMPT_TEMPLATE = ""


class PairwiseComparison:

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
        doc1: str,
        doc2: str,
    ) -> str:
        # TODO
        input_context = f"Given a query: {query}, which of the following two passages is more relevant to the query?\n\nPassage A: {doc1}\n\nPassage B: {doc2}\n\nOutput Passage A or Passage B."
        messages = [{"role": "user", "content": input_context}]
        return messages

    def __call__(self, query: str, doc1: str, doc2: str, **kwargs) -> int:
        messages = self.create_messages(query, doc1, doc2) + self.create_messages(query, doc2, doc1)
        outputs = self.lm.generate(
            messages,
            **kwargs,
        )
        # TODO: parse the output
        if outputs[0] == "passage a" and outputs[1] == "passage b":
            score = 1
        elif outputs[0] == "passage a" and outputs[1] == "passage b":
            score = -1
        else:
            score = 0
        return score
