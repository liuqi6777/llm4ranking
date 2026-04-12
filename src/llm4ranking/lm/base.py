from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class LMOutput:
    text: Optional[str] = None
    loglikelihood: Optional[float] = None
    logits: Optional[np.ndarray] = None


@dataclass
class BatchLMOutput:
    text: Optional[list[str]] = None
    loglikelihood: Optional[list[float]] = None
    logits: Optional[list[Union[np.ndarray, float, list[float]]]] = None


class LM(ABC):
    supports_batch_generate = False
    supports_batch_loglikelihood = False
    supports_batch_logits = False

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, messages: dict[str, str], **kwargs) -> Union[str, LMOutput]:
        pass

    @abstractmethod
    def loglikelihood(self, messages: dict[str, str], **kwargs) -> Union[float, LMOutput]:
        pass

    @abstractmethod
    def logits(self, messages: dict[str, str], **kwargs) -> Union[np.ndarray, LMOutput]:
        pass

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        outputs = [self.generate(messages, **kwargs) for messages in batch_messages]
        return BatchLMOutput(
            text=[output.text for output in outputs],
        )

    def loglikelihood_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        outputs = [self.loglikelihood(messages, **kwargs) for messages in batch_messages]
        return BatchLMOutput(
            text=[output.text for output in outputs],
            loglikelihood=[output.loglikelihood for output in outputs],
        )

    def logits_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        token: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> BatchLMOutput:
        outputs = [self.logits(messages, token=token, **kwargs) for messages in batch_messages]
        return BatchLMOutput(
            logits=[output.logits for output in outputs],
        )
