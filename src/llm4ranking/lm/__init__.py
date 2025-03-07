from llm4ranking.lm.huggingface import HFLM
from llm4ranking.lm.openai import OpenAIClient


def load_model(
    model_type: str,
    model_args: dict,
):
    if model_type == "hf":
        return HFLM(**model_args)
    elif model_type == "openai":
        return OpenAIClient(**model_args)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
