from llm4ranking.model.lm import create_model


class QueryGeneration:

    def __init__(self):
        self.lm = create_model()  # FIXME

    def create_messages(
        self,
        query: str,
        doc: str,
    ) -> str:
        input_context = f"Passage: {doc}\nPlease write a question based on this passage."
        messages = [{"role": "user", "content": input_context}, {"role": "assistant", "content": query}]
        return messages

    def __call__(self, query: str, doc: str) -> float:
        messages = self.create_messages(query, doc)
        score = self.lm.loglikelihood(messages)
        return score
