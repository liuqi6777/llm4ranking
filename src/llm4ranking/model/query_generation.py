from llm4ranking.model.base import BaseRankingModel


DEFAULT_PROMPT_TEMPLATE = """Please write a question based on this document.

Document: {{ doc }}

Query:"""


class QueryGeneration(BaseRankingModel):

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        messages = [
            {"role": "user", "content": self.prompt_template.render(doc=doc)},
            {"role": "assistant", "content": query}
        ]
        return messages

    def __call__(self, query: str, doc: str) -> float:
        messages = self.create_messages(query, doc)
        score = self.lm.loglikelihood(messages)
        return score
