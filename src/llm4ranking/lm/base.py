from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
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


class Capability(Enum):
    GENERATE = auto()
    LOGLIKELIHOOD = auto()
    LOGITS = auto()
    BATCH_GENERATE = auto()
    BATCH_LOGLIKELIHOOD = auto()
    BATCH_LOGITS = auto()


class LM(ABC):
    supports_batch_generate = False
    supports_batch_loglikelihood = False
    supports_batch_logits = False
    capabilities: set[Capability] = set()

    def __init__(self, **kwargs):
        pass

    def _effective_capabilities(self) -> set[Capability]:
        capabilities = set(self.capabilities)
        if type(self).generate is not LM.generate:
            capabilities.add(Capability.GENERATE)
            capabilities.add(Capability.BATCH_GENERATE)
        if type(self).loglikelihood is not LM.loglikelihood:
            capabilities.add(Capability.LOGLIKELIHOOD)
            capabilities.add(Capability.BATCH_LOGLIKELIHOOD)
        if type(self).logits is not LM.logits:
            capabilities.add(Capability.LOGITS)
            capabilities.add(Capability.BATCH_LOGITS)
        return capabilities

    def has_capabilities(self, *required: Capability) -> bool:
        return set(required).issubset(self._effective_capabilities())

    def require_capabilities(self, *required: Capability) -> None:
        missing = set(required) - self._effective_capabilities()
        if missing:
            missing_names = ", ".join(capability.name for capability in sorted(missing, key=lambda item: item.name))
            raise ValueError(
                f"{self.__class__.__name__} does not support required capabilities: {missing_names}"
            )

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs) -> Union[str, LMOutput]:
        pass

    @abstractmethod
    def loglikelihood(self, messages: list[dict[str, str]], **kwargs) -> Union[float, LMOutput]:
        pass

    @abstractmethod
    def logits(self, messages: list[dict[str, str]], **kwargs) -> Union[np.ndarray, LMOutput]:
        pass

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        self.require_capabilities(Capability.GENERATE)
        outputs = [self.generate(messages, **kwargs) for messages in batch_messages]
        return BatchLMOutput(
            text=[output.text for output in outputs],
        )

    def loglikelihood_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs,
    ) -> BatchLMOutput:
        self.require_capabilities(Capability.LOGLIKELIHOOD)
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
        self.require_capabilities(Capability.LOGITS)
        outputs = [self.logits(messages, token=token, **kwargs) for messages in batch_messages]
        return BatchLMOutput(
            logits=[output.logits for output in outputs],
        )
