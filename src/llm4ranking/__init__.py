from typing import Optional, Union, List, Dict
from functools import partial

from llm4ranking.ranker import *
from llm4ranking.model import *


RERANKING_APPROACHES = {
    "rankgpt": (ListwiseSilidingWindowReranker, RankGPT),
    "rel-gen": (PointwiseReranker, RelevanceGeneration),
    "query-gen": (PointwiseReranker, QueryGeneration),
    "prp": (PairwiseHeapSortReranker, PRP),
    "tourrank": (TournamentReranker, Selection),
    "first": (ListwiseSilidingWindowReranker, First),
    "fg-rel-gen": (PointwiseReranker, FineGrainedRelevanceGeneration),
}


def get_default_args_by_approach(approach: str) -> Dict:
    """Get default arguments for each reranking approach.

    Args:
        approach (str): The reranking approach.

    Returns:
        Dict: Default arguments for the specified approach.
    """
    # TODO: Add more default arguments
    defaults = {
        "rankgpt": {},
        "rel-gen": {},
        "query-gen": {},
        "prp": {},
        "tourrank": {}
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
        default_model_fw_args = {
            # "max_length": default_reranking_args.pop("max_length", 1024),
            "temperature": default_reranking_args.pop("temperature", 0.0),
        }

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
        reranking_args = {**default_reranking_args, **reranking_args}
        model_fw_args = {**default_model_fw_args, **model_fw_args}

        self.reranker = RERANKING_APPROACHES[reranking_approach][0]()
        self.ranking_func = RERANKING_APPROACHES[reranking_approach][1](model_type, model_args, prompt_template)
        self.rerank = partial(
            self.reranker.rerank,
            ranking_func=self.ranking_func,
            **reranking_args,
            **model_fw_args
        )

    def rerank(
        self,
        query: str,
        candidates: List[str]
    ) -> List[int]:
        """Rerank a list of candidates given a query.
        """
        pass

