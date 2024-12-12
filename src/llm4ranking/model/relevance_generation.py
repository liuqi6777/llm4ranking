from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """Document: {{ doc }}

Query: {{ query }}

Does the document answer the query?
"""


class RelevanceGeneration(BaseRankingModel):

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        messages = [
            {"role": "user", "content": self.prompt_template.render(doc=doc, query=query)},
            {"role": "assistant", "content": " Yes"}
        ]
        return messages

    def __call__(self, query: str, doc: str) -> float:
        messages = self.create_messages(query, doc)
        score = self.lm.loglikelihood(messages)
        return score
