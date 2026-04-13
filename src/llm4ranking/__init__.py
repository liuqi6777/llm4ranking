from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm4ranking.config import (
    BackendRuntimeArgs,
    ModelConfig,
    RerankerConfig,
    StrategyRuntimeArgs,
)
from llm4ranking.policy import *
from llm4ranking.strategy import *


@dataclass(frozen=True)
class ApproachSpec:
    strategy_cls: type[RerankStrategy]
    policy_cls: type[BaseRankingPolicy]
    default_strategy_args: dict = field(default_factory=dict)
    allowed_strategy_args: frozenset[str] = field(default_factory=frozenset)


BACKEND_ARG_NAMES = frozenset(BackendRuntimeArgs.__dataclass_fields__.keys())

DEFAULT_ALLOWED_STRATEGY_ARGS_BY_CLASS = {
    Pointwise: frozenset({"return_record", "truncate_length", "batch_size"}),
    PairwiseAllPair: frozenset({"return_record", "pair_batch_size"}),
    PairwiseHeapSort: frozenset({"return_record", "topk"}),
    PairwiseBubbleSort: frozenset({"return_record", "topk"}),
    ListwiseSlidingWindow: frozenset({"return_record", "rank_start", "rank_end", "window_size", "step", "truncate_length"}),
    Tournament: frozenset(
        {"return_record", "tournament_times", "group_sizes", "promotion_sizes", "stage_weight", "truncate_length"}
    ),
}


RERANKING_APPROACHES = {
    "rankgpt": ApproachSpec(
        strategy_cls=ListwiseSlidingWindow,
        policy_cls=RankGPT,
        default_strategy_args={"window_size": 20, "step": 10, "truncate_length": 300},
        allowed_strategy_args=frozenset({"return_record", "rank_start", "rank_end", "window_size", "step", "truncate_length"}),
    ),
    "rel-gen": ApproachSpec(
        strategy_cls=Pointwise,
        policy_cls=RelevanceGeneration,
        allowed_strategy_args=frozenset({"return_record", "truncate_length", "batch_size"}),
    ),
    "query-gen": ApproachSpec(
        strategy_cls=Pointwise,
        policy_cls=QueryGeneration,
        allowed_strategy_args=frozenset({"return_record", "truncate_length", "batch_size"}),
    ),
    "prp-heap": ApproachSpec(
        strategy_cls=PairwiseHeapSort,
        policy_cls=PRP,
        default_strategy_args={"topk": 10},
        allowed_strategy_args=frozenset({"return_record", "topk"}),
    ),
    "prp-allpair": ApproachSpec(
        strategy_cls=PairwiseAllPair,
        policy_cls=PRP,
        allowed_strategy_args=frozenset({"return_record", "pair_batch_size"}),
    ),
    "prp-bubble": ApproachSpec(
        strategy_cls=PairwiseBubbleSort,
        policy_cls=PRP,
        default_strategy_args={"topk": 10},
        allowed_strategy_args=frozenset({"return_record", "topk"}),
    ),
    "tourrank": ApproachSpec(
        strategy_cls=Tournament,
        policy_cls=TourRankSelection,
        default_strategy_args={"tournament_times": 1, "truncate_length": 300},
        allowed_strategy_args=frozenset(
            {"return_record", "tournament_times", "group_sizes", "promotion_sizes", "stage_weight", "truncate_length"}
        ),
    ),
    "first": ApproachSpec(
        strategy_cls=ListwiseSlidingWindow,
        policy_cls=First,
        default_strategy_args={"window_size": 20, "step": 10, "truncate_length": 300},
        allowed_strategy_args=frozenset({"return_record", "rank_start", "rank_end", "window_size", "step", "truncate_length"}),
    ),
    "fg-rel-gen": ApproachSpec(
        strategy_cls=Pointwise,
        policy_cls=FineGrainedRelevanceGeneration,
        allowed_strategy_args=frozenset({"return_record", "truncate_length", "batch_size"}),
    ),
}


def get_approach_spec(approach: str) -> ApproachSpec:
    value = RERANKING_APPROACHES[approach]
    if isinstance(value, ApproachSpec):
        return value

    strategy_cls, policy_cls = value
    return ApproachSpec(
        strategy_cls=strategy_cls,
        policy_cls=policy_cls,
        allowed_strategy_args=DEFAULT_ALLOWED_STRATEGY_ARGS_BY_CLASS.get(strategy_cls, frozenset()),
    )


def list_available_reranking_approaches() -> List[str]:
    return list(RERANKING_APPROACHES.keys())


def get_default_args_by_approach(approach: str) -> Dict:
    if approach not in RERANKING_APPROACHES:
        return {}
    return dict(get_approach_spec(approach).default_strategy_args)


class Reranker:
    """User-facing reranking facade."""

    def __init__(
        self,
        config: Optional[RerankerConfig] = None,
        reranking_approach: str = "rankgpt",
        model_type: str = "openai",
        model_name: Optional[str] = None,
        model_args: Optional[dict] = None,
        strategy_args: Optional[dict] = None,
        backend_args: Optional[dict] = None,
        prompt_template: Optional[str] = None,
    ):
        (
            reranking_approach,
            model_type,
            model_name,
            model_args,
            strategy_args,
            backend_args,
            prompt_template,
        ) = self._normalize_init_args(
            config=config,
            reranking_approach=reranking_approach,
            model_type=model_type,
            model_name=model_name,
            model_args=model_args,
            strategy_args=strategy_args,
            backend_args=backend_args,
            prompt_template=prompt_template,
        )

        if reranking_approach not in RERANKING_APPROACHES:
            raise ValueError(
                f"Unsupported reranking approach: {reranking_approach}. "
                f"Available approaches: {list_available_reranking_approaches()}"
            )

        if model_args is None:
            if model_name is None:
                raise ValueError("model_name must be provided if model_args is None")
            model_args = {"model": model_name}
        else:
            model_args = dict(model_args)
            if "model" not in model_args and model_name is None:
                raise ValueError("model_args must contain 'model' key or model_name must be provided")
            model_args["model"] = model_args.get("model", model_name)

        self.reranking_approach = reranking_approach
        self._strategy_kwargs = {
            **get_default_args_by_approach(reranking_approach),
            **dict(strategy_args or {}),
        }
        self._validate_strategy_args(self._strategy_kwargs)
        self._backend_kwargs = dict(backend_args or {})
        self.config = RerankerConfig(
            reranking_approach=reranking_approach,
            model=ModelConfig(
                model_type=model_type,
                model_name=model_args["model"],
                init_args=dict(model_args),
                inference_args=dict(self._backend_kwargs),
            ),
            strategy_args=dict(self._strategy_kwargs),
            prompt_template=prompt_template,
        )

        approach_spec = get_approach_spec(reranking_approach)
        self.strategy = approach_spec.strategy_cls()
        self.policy = approach_spec.policy_cls(model_type, model_args, prompt_template)

    @classmethod
    def from_config(cls, config: RerankerConfig) -> "Reranker":
        return cls(config=config)

    def rerank(
        self,
        query: str,
        candidates: List[str],
        strategy: Optional[StrategyRuntimeArgs] = None,
        backend: Optional[BackendRuntimeArgs] = None,
        **kwargs,
    ) -> RerankResult:
        rerank_kwargs = self._build_rerank_kwargs(kwargs, strategy=strategy, backend=backend)
        return self.strategy.rerank(
            query=query,
            candidates=candidates,
            ranking_func=self.policy,
            **rerank_kwargs,
        )

    def _build_rerank_kwargs(
        self,
        override_kwargs: dict,
        *,
        strategy: Optional[StrategyRuntimeArgs] = None,
        backend: Optional[BackendRuntimeArgs] = None,
    ) -> dict:
        strategy_runtime_kwargs = strategy.as_kwargs() if strategy is not None else {}
        backend_runtime_kwargs = backend.as_kwargs() if backend is not None else {}
        override_strategy_kwargs = {
            key: value for key, value in override_kwargs.items()
            if key not in BACKEND_ARG_NAMES
        }
        self._validate_strategy_args({
            **override_strategy_kwargs,
            **strategy_runtime_kwargs,
        })
        return {
            **self._strategy_kwargs,
            **self._backend_kwargs,
            **override_kwargs,
            **strategy_runtime_kwargs,
            **backend_runtime_kwargs,
        }

    def _validate_strategy_args(self, strategy_args: dict) -> None:
        spec = get_approach_spec(self.reranking_approach)
        invalid_args = sorted(set(strategy_args) - set(spec.allowed_strategy_args))
        if not invalid_args:
            return

        allowed_args = ", ".join(sorted(spec.allowed_strategy_args))
        invalid_args_display = ", ".join(invalid_args)
        raise ValueError(
            f"Unsupported strategy args for approach '{self.reranking_approach}': {invalid_args_display}. "
            f"Allowed args: {allowed_args}"
        )

    @staticmethod
    def _normalize_init_args(
        *,
        config: Optional[RerankerConfig],
        reranking_approach: str,
        model_type: str,
        model_name: Optional[str],
        model_args: Optional[dict],
        strategy_args: Optional[dict],
        backend_args: Optional[dict],
        prompt_template: Optional[str],
    ):
        normalized_strategy_args = dict(strategy_args or {})
        normalized_backend_args = dict(backend_args or {})

        if config is None:
            return (
                reranking_approach,
                model_type,
                model_name,
                model_args,
                normalized_strategy_args,
                normalized_backend_args,
                prompt_template,
            )

        if any(
            value is not None
            for value in (model_args, strategy_args, backend_args)
        ):
            raise ValueError("When `config` is provided, do not also pass explicit args dictionaries.")
        if model_name is not None or prompt_template is not None:
            raise ValueError("When `config` is provided, do not also pass `model_name` or `prompt_template`.")
        if reranking_approach != "rankgpt" or model_type != "openai":
            raise ValueError("When `config` is provided, do not also override `reranking_approach` or `model_type`.")

        model_init_args = dict(config.model.init_args)
        if "model" not in model_init_args:
            if config.model.model_name is None:
                raise ValueError("`config.model.model_name` or `config.model.init_args['model']` must be provided.")
            model_init_args["model"] = config.model.model_name

        return (
            config.reranking_approach,
            config.model.model_type,
            model_init_args["model"],
            model_init_args,
            dict(config.strategy_args),
            dict(config.model.inference_args),
            config.prompt_template,
        )
