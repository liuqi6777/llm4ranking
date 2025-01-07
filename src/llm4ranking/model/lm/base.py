from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class LMOuput:
    text: str
    loglikelihood: Optional[float] = None
    num_processed_tokens: Optional[int] = None
    num_generated_tokens: Optional[int] = None


class LM(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, messages: dict[str, str], **kwargs) -> Union[str, LMOuput]:
        pass

    @abstractmethod
    def loglikelihood(self, messages: dict[str, str], **kwargs) -> Union[float, LMOuput]:
        pass
