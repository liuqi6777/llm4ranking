from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """Given a query: {{ query }}, which of the following two documents is more relevant to the query?

Document A: {{ doc1 }}

Document B: {{ doc2 }}

Only output "A" or "B", do not say anything else or explain.
"""


class PairwiseComparison(BaseRankingModel):

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def __call__(self, query: str, doc1: str, doc2: str, **kwargs) -> int:
        outputs = []
        outputs.append(self.lm.generate(self.create_messages(query, doc1, doc2), **kwargs))
        outputs.append(self.lm.generate(self.create_messages(query, doc2, doc1), **kwargs))
        if self.parse_output(outputs[0]) == "a" and self.parse_output(outputs[1]) == "b":
            score = 1
        elif self.parse_output(outputs[0]) == "b" and self.parse_output(outputs[1]) == "a":
            score = -1
        else:
            score = 0
        return score

    def create_messages(
        self,
        query: str,
        doc1: str,
        doc2: str,
    ) -> str:
        messages = [
            {"role": "user", "content": self.prompt_template.render(query=query, doc1=doc1, doc2=doc2)}
        ]
        return messages

    def parse_output(self, output: str) -> str:
        return output.strip().lower()[0]
