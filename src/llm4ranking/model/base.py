from abc import ABC, abstractmethod
from jinja2 import Template
from typing import Optional

from llm4ranking.model.lm import load_model


class BaseRankingModel(ABC):

    DEFAULT_PROMPT_TEMPLATE = ""

    def __init__(
        self,
        model_type: str,
        model_args: dict,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        self.lm = load_model(model_type, model_args)
        self._prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @property
    def prompt_template(self):
        return Template(self._prompt_template)

    @prompt_template.setter
    def prompt_template(self, value):
        self._prompt_template = Template(value)
