def load_model(
    model_type: str,
    model_args: dict,
):
    if model_type == "hf":
        from llm4ranking.lm.huggingface import HFLM

        return HFLM(**model_args)
    elif model_type == "openai":
        from llm4ranking.lm.openai import OpenAIClient

        return OpenAIClient(**model_args)
    elif model_type == "vllm":
        try:
            from llm4ranking.lm.vllm import VLLM
        except Exception as exc:
            raise RuntimeError("vllm backend is unavailable.") from exc
        return VLLM(**model_args)
    else:
        raise ValueError(f"Model type {model_type} is not supported.")
