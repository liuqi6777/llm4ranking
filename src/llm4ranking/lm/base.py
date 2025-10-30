from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


@dataclass
class LMOutput:
    text: Optional[str] = None
    loglikelihood: Optional[float] = None
    logits: Optional[np.ndarray] = None


class LM(ABC):
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
