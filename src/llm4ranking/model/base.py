from abc import ABC, abstractmethod
from jinja2 import Template
from typing import Optional

from llm4ranking.lm import load_model


class BaseRankingModel(ABC):
    """Base class for all ranking models.
    
    This abstract class defines the interface that all ranking models must implement.
    It handles loading the language model and managing prompt templates.
    """

    DEFAULT_PROMPT_TEMPLATE = ""

    ranker = "base"
    name = ""

    def __init__(
        self,
        model_type: str,
        model_args: dict,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the ranking model.

        Args:
            model_type (str): Type of language model to use ('hf' or 'openai')
            model_args (dict): Arguments for initializing the language model
            prompt_template (Optional[str], optional): Custom prompt template. Defaults to None.
            **kwargs: Additional keyword arguments
        """
        self.lm = load_model(model_type, model_args)
        self._prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    @abstractmethod
    def __call__(self, **kwargs):
        """Execute the ranking operation.
        
        This method must be implemented by all subclasses.
        """
        pass

    @property
    def prompt_template(self):
        """Get the Jinja2 template for generating prompts."""
        return Template(self._prompt_template)

    @prompt_template.setter
    def prompt_template(self, value):
        """Set a new prompt template.

        Args:
            value (str): New template string
        """
        self._prompt_template = Template(value)

    @property
    def ranker(self):
        """Get the ranker type."""
        return self.ranker
