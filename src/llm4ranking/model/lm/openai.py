from llm4ranking.model.lm.base import LM


class OpenAILM(LM):
    def __init__(self, **kwargs):
        pass

    def generate(self, messages, **kwargs):
        pass

    def loglikelihood(self, messages, **kwargs):
        raise NotImplementedError
    