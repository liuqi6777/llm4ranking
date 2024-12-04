from llm4ranking.model.lm import create_model


class RelevanceGeneration:

    def __init__(self):
        self.lm = create_model()  # FIXME

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
