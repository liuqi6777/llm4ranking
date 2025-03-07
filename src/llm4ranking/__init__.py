from llm4ranking.evaluation.evaluator import simple_evaluate, simple_rerank
from typing import Optional, Union, List, Dict


def get_default_args_by_approach(approach: str) -> Dict:
    """Get default arguments for each reranking approach.

    Args:
        approach (str): The reranking approach.

    Returns:
        Dict: Default arguments for the specified approach.
    """
    defaults = {
        "listwise-sw": {},
        "pointwise-rg": {},
        "pointwise-qg": {},
        "pairwise": {},
        "tournament": {}
    }
    return defaults.get(approach, {})


def rerank(
    query: str,
    candidates: List[str],
    model: str = "meta-llama/Llama-2-7b-chat-hf",
    approach: str = "listwise-sw",
    model_type: str = "hf",
    model_args: Optional[dict] = None,
    reranking_args: Optional[dict] = None,
    model_fw_args: Optional[dict] = None,
    prompt_template: Optional[str] = None,
) -> Union[List[str], List[int]]:
    """Easy-to-use function for reranking a list of candidates given a query.

    This is the main entry point for using the library. It provides a simple interface
    to rerank documents using various approaches and models.

    Args:
        query (str): The search query.
        candidates (List[str]): List of candidate documents/passages to rerank.
        model (str, optional): Model identifier or path. Defaults to "meta-llama/Llama-2-7b-chat-hf".
        approach (str, optional): Reranking approach. One of ["listwise-sw", "pointwise-rg", "pointwise-qg", "pairwise", "tournament"]. 
            Defaults to "listwise-sw".
        model_type (str, optional): Type of model to use. One of ["hf", "openai"]. Defaults to "hf".
        model_args (dict, optional): Additional arguments for model initialization. Defaults to None.
        reranking_args (dict, optional): Additional arguments for reranking. Defaults to None.
        model_fw_args (dict, optional): Additional arguments for model forward pass. Defaults to None.
        prompt_template (str, optional): Custom prompt template. Defaults to None.

    Returns:
        Union[List[str], List[int]]: Reranked candidates or indices of reranked candidates.
    """
    # Get default arguments for the specified approach
    default_reranking_args = get_default_args_by_approach(approach)
    default_model_fw_args = {
        "max_length": default_reranking_args.pop("max_length", 1024),
        "temperature": default_reranking_args.pop("temperature", 0.0),
    }

    # Initialize arguments with defaults
    if model_args is None:
        model_args = {"model": model}
    if reranking_args is None:
        reranking_args = {}
    if model_fw_args is None:
        model_fw_args = {}

    # Merge defaults with user-provided arguments
    reranking_args = {**default_reranking_args, **reranking_args}
    model_fw_args = {**default_model_fw_args, **model_fw_args}

    reranked_indices = simple_rerank(
        query=query,
        candidates=candidates,
        reranking_approach=approach,
        model_type=model_type,
        model_args=model_args,
        reranking_args=reranking_args,
        model_fw_args=model_fw_args,
        prompt_template=prompt_template,
    )

    return [candidates[i] for i in reranked_indices]

