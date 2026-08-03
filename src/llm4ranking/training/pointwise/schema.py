from typing import Any


def passage_to_text(passage: Any) -> str:
    if isinstance(passage, str):
        return passage
    if isinstance(passage, dict):
        title = passage.get("title")
        text = passage.get("text", passage.get("content"))
        if text is None:
            raise ValueError("Passage dictionaries must contain a 'text' or 'content' field.")
        return f"{title}\n{text}" if title and title != "-" else str(text)
    raise TypeError(f"Passages must be strings or dictionaries, got {type(passage).__name__}.")


def normalize_passages(value: Any, field_name: str) -> list[str]:
    if value is None:
        raise ValueError(f"Training sample is missing '{field_name}'.")
    passages = value if isinstance(value, list) else [value]
    normalized = [passage_to_text(passage) for passage in passages]
    if not normalized:
        raise ValueError(f"Training sample contains no passages in '{field_name}'.")
    return normalized


def normalize_pointwise_sample(sample: dict) -> tuple[str, list[str], list[str]]:
    if "query" not in sample:
        raise ValueError("Training sample is missing 'query'.")
    positives = normalize_passages(
        sample.get("positive", sample.get("positive_passages")),
        "positive/positive_passages",
    )
    negatives = normalize_passages(
        sample.get("negative", sample.get("negative_passages")),
        "negative/negative_passages",
    )
    return str(sample["query"]), positives, negatives
