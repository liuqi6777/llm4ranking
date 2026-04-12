from abc import ABC, abstractmethod
from typing import Optional

from jinja2 import Template

from llm4ranking.lm import load_model
from llm4ranking.lm.base import BatchLMOutput, Capability, LM, LMOutput


class BaseRankingPolicy(ABC):
    """Base class for ranking-oriented prompt adapters.

    The ranking model sits between the reranking strategy and the LM backend:
    it defines prompt construction, output parsing, and any scoring/comparison logic.
    """

    DEFAULT_PROMPT_TEMPLATE = ""
    model_kind = "base"
    name = ""
    supports_batch = False
    required_capabilities: set[Capability] = set()

    def __init__(
        self,
        model_type: str,
        model_args: dict,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        self.lm = load_model(model_type, model_args)
        self._prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.validate_backend()

    def validate_backend(self) -> None:
        if hasattr(self.lm, "require_capabilities"):
            self.lm.require_capabilities(*self.required_capabilities)
            return

        backend_capabilities = getattr(self.lm, "capabilities", set())
        missing = self.required_capabilities - set(backend_capabilities)
        if missing:
            missing_names = ", ".join(capability.name for capability in sorted(missing, key=lambda item: item.name))
            raise ValueError(
                f"{self.lm.__class__.__name__} does not support required capabilities: {missing_names}"
            )

    @abstractmethod
    def __call__(self, **kwargs):
        """Execute the ranking operation."""
        pass

    def create_batch_messages(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement batch message creation.")

    def parse_batch_outputs(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement batch output parsing.")

    def _unwrap_batch_output(self, batch_output: BatchLMOutput) -> LMOutput:
        return LMOutput(
            text=batch_output.text[0] if batch_output.text else None,
            loglikelihood=batch_output.loglikelihood[0] if batch_output.loglikelihood else None,
            logits=batch_output.logits[0] if batch_output.logits else None,
        )

    @property
    def prompt_template(self) -> Template:
        return Template(self._prompt_template)

    @prompt_template.setter
    def prompt_template(self, value: str) -> None:
        self._prompt_template = value

    @property
    def backend(self) -> LM:
        return self.lm


class PointwisePolicy(BaseRankingPolicy):
    model_kind = "pointwise"

    @abstractmethod
    def score_many(self, query: str, docs: list[str], **kwargs):
        pass


class PairwisePolicy(BaseRankingPolicy):
    model_kind = "pairwise"

    @abstractmethod
    def compare_many(self, query: str, doc_pairs: list[tuple[str, str]], **kwargs):
        pass


class ListwisePolicy(BaseRankingPolicy):
    model_kind = "listwise"


class SelectionPolicy(BaseRankingPolicy):
    model_kind = "selection"
