from abc import ABC, abstractmethod
from typing import Any


class RankingModel(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
