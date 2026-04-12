from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_type: str = "openai"
    model_name: Optional[str] = None
    init_args: dict = field(default_factory=dict)
    inference_args: dict = field(default_factory=dict)


@dataclass
class StrategyRuntimeArgs:
    return_record: Optional[bool] = None

    truncate_length: Optional[int] = None

    batch_size: Optional[int] = None

    pair_batch_size: Optional[int] = None
    topk: Optional[int] = None

    rank_start: Optional[int] = None
    rank_end: Optional[int] = None
    window_size: Optional[int] = None
    step: Optional[int] = None

    tournament_times: Optional[int] = None
    group_sizes: Optional[tuple[int, ...]] = None
    promotion_sizes: Optional[tuple[int, ...]] = None
    stage_weight: Optional[float] = None

    def as_kwargs(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class BackendRuntimeArgs:
    max_new_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None

    def as_kwargs(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class RerankerConfig:
    reranking_approach: str = "rankgpt"
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy_args: dict = field(default_factory=dict)
    prompt_template: Optional[str] = None
