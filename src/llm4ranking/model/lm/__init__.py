from llm4ranking.model.lm.huggingface import HFLM
from llm4ranking.model.lm.openai import OpenAILM


def load_model(
    model_type: str,
    model_args: dict,
):
    if model_type == "hf":
        return HFLM(**model_args)
    elif model_type == "openai":
        raise OpenAILM(**model_args)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
