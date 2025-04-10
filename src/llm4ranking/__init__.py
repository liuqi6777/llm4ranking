from typing import Optional, List, Dict
from functools import partial

from llm4ranking.ranker import *
from llm4ranking.model import *


RERANKING_APPROACHES = {
    "rankgpt": (ListwiseSilidingWindowReranker, RankGPT),
    "rel-gen": (PointwiseReranker, RelevanceGeneration),
    "query-gen": (PointwiseReranker, QueryGeneration),
    "prp-heap": (PairwiseHeapSortReranker, PRP),
    "prp-allpair": (PairwiseAllPairReranker, PRP),
    "prp-bubble": (PairwiseBubbleSortReranker, PRP),
    "tourrank": (TournamentReranker, TourRankSelection),
    "first": (ListwiseSilidingWindowReranker, First),
    "fg-rel-gen": (PointwiseReranker, FineGrainedRelevanceGeneration),
}


def list_available_reranking_approaches() -> List[str]:
    """List available reranking approaches.

    Returns:
        List[str]: List of available reranking approaches.
    """
    return list(RERANKING_APPROACHES.keys())


def get_default_args_by_approach(approach: str) -> Dict:
    """Get default arguments for each reranking approach.

    Args:
        approach (str): The reranking approach.

    Returns:
        Dict: Default arguments for the specified approach.
    """
    # TODO: Add more default arguments
    defaults = {
        "rankgpt": {
            "window_size": 20, "step": 10, "truncate_length": 300,
            "do_sample": False, "max_new_tokens": 120
        },
        "rel-gen": {},
        "query-gen": {},
        "prp": {"topk": 10},
        "tourrank": {
            "tuornament_times": 1, "truncate_length": 300,
            "do_sample": False, "max_new_tokens": 120
        },
        "first": {
            "window_size": 20, "step": 10, "truncate_length": 300,
        }
    }
    return defaults.get(approach, {})


class Reranker:
    """Easy-to-use class for reranking a list of candidates given a query.

    This is the main entry point for using the library. It provides a simple interface
    to rerank documents using various approaches and models.
    """

    def __init__(
        self,
        reranking_approach: str = "rankgpt",
        model_type: str = "openai",
        model_name: Optional[str] = None,
        model_args: Optional[dict] = None,
        reranking_args: Optional[dict] = None,
        model_fw_args: Optional[dict] = None,
        prompt_template: Optional[str] = None,
    ):
        """Initialize the reranker.
        Args:
            reranking_approach (str, optional): Reranking approach.
            model_type (str, optional): Type of model to use. One of ["hf", "openai", "vllm"]. Defaults to "openai".
            model_name (str, optional): Model identifier or path.
            model_args (dict, optional): Additional arguments for model initialization.
            reranking_args (dict, optional): Additional arguments for reranking.
            model_fw_args (dict, optional): Additional arguments for model forward pass.
            prompt_template (str, optional): Custom prompt template.

        Returns:
            Tuple[List[str], List[int]]: Reranked candidates and indices of reranked candidates.
        """
        default_reranking_args = get_default_args_by_approach(reranking_approach)

        # Initialize arguments with defaults
        if model_args is None:
            assert model_name is not None, "model_name must be provided if model_args is None"
            model_args = {"model": model_name}
        else:
            if "model" not in model_args and model_name is None:
                raise ValueError("model_args must contain 'model' key or model_name must be provided")
            model_args["model"] = model_args.get("model", model_name)
        if reranking_args is None:
            reranking_args = {}
        if model_fw_args is None:
            model_fw_args = {}

        # Merge defaults with user-provided arguments
        args = {**default_reranking_args, **reranking_args, **model_fw_args}

        self.ranker = RERANKING_APPROACHES[reranking_approach][0]()
        self.ranking_func = RERANKING_APPROACHES[reranking_approach][1](model_type, model_args, prompt_template)
        self.rerank = partial(
            self.ranker.rerank,
            ranking_func=self.ranking_func,
            **args
        )

    def rerank(
        self,
        query: str,
        candidates: List[str],
        **kwargs
    ) -> List[int]:
        """Rerank a list of candidates given a query.
        """
        pass

    @property
    def LLM(self):
        return self.ranking_func.lm
