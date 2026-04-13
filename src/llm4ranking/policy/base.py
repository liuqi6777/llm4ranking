from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from jinja2 import Template

from llm4ranking.lm import load_model
from llm4ranking.lm.base import BatchLMOutput, Capability, LM, LMOutput


T = TypeVar("T")


@dataclass(frozen=True)
class PolicyResult(Generic[T]):
    value: T
    lm_outputs: Any = None


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

    def __call__(self, **kwargs):
        """Execute the ranking operation."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `__call__`.")

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

    def _wrap_lm_outputs(self, lm_outputs: list[Any]) -> BatchLMOutput:
        return BatchLMOutput(
            text=[getattr(output, "text", None) for output in lm_outputs],
            loglikelihood=[getattr(output, "loglikelihood", None) for output in lm_outputs],
            logits=[getattr(output, "logits", None) for output in lm_outputs],
        )

    def make_result(self, value: T, lm_outputs: Any) -> PolicyResult[T]:
        return PolicyResult(value=value, lm_outputs=lm_outputs)

    def normalize_result(self, result: Any) -> PolicyResult[Any]:
        if isinstance(result, PolicyResult):
            return result
        return PolicyResult(value=result, lm_outputs=None)

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
    def score(self, query: str, doc: str, **kwargs):
        pass

    def __call__(self, query: str, doc: str, **kwargs):
        return self.score(query, doc, **kwargs)

    def score_many(self, query: str, docs: list[str], return_lm_outputs: bool = False, **kwargs):
        if return_lm_outputs:
            scored_outputs = [
                self.normalize_result(self.score(query, doc, return_lm_outputs=True, **kwargs))
                for doc in docs
            ]
            scores = [result.value for result in scored_outputs]
            lm_outputs = [result.lm_outputs for result in scored_outputs]
            return self.make_result(scores, self._wrap_lm_outputs(lm_outputs))

        return [self.score(query, doc, **kwargs) for doc in docs]

    def score_batch(self, *args, **kwargs):
        return self.score_many(*args, **kwargs)


class PairwisePolicy(BaseRankingPolicy):
    model_kind = "pairwise"

    @abstractmethod
    def compare(self, query: str, doc1: str, doc2: str, **kwargs):
        pass

    def __call__(self, query: str, doc1: str, doc2: str, **kwargs):
        return self.compare(query, doc1, doc2, **kwargs)

    def compare_many(self, query: str, doc_pairs: list[tuple[str, str]], return_lm_outputs: bool = False, **kwargs):
        if return_lm_outputs:
            compared_outputs = [
                self.normalize_result(self.compare(query, doc1, doc2, return_lm_outputs=True, **kwargs))
                for doc1, doc2 in doc_pairs
            ]
            scores = [result.value for result in compared_outputs]
            lm_outputs = [result.lm_outputs for result in compared_outputs]
            return self.make_result(scores, lm_outputs)

        return [self.compare(query, doc1, doc2, **kwargs) for doc1, doc2 in doc_pairs]

    def compare_batch(self, *args, **kwargs):
        return self.compare_many(*args, **kwargs)


class ListwisePolicy(BaseRankingPolicy):
    model_kind = "listwise"

    @abstractmethod
    def rank(self, query: str, candidates: list[str], **kwargs):
        pass

    def __call__(self, query: str, candidates: list[str], **kwargs):
        return self.rank(query, candidates, **kwargs)


class SelectionPolicy(BaseRankingPolicy):
    model_kind = "selection"

    @abstractmethod
    def select(self, query: str, candidates: list[str], num_selection: int, **kwargs):
        pass

    def __call__(self, query: str, candidates: list[str], num_selection: int, **kwargs):
        return self.select(query, candidates, num_selection, **kwargs)
