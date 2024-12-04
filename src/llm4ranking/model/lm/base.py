from abc import ABC, abstractmethod


class LM(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, messages: dict[str, str], **kwargs) -> str:
        pass

    @abstractmethod
    def loglikelihood(self, messages: dict[str, str], **kwargs) -> float:
        pass
