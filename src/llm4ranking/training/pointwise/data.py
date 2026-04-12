import ujson as json
from typing import Dict, List, Sequence, Any
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


PROMPT = """Document: {document}

Query: {query}

Does the document answer the query?"""


class PointwiseDataset(Dataset):
    def __init__(self, data_path: str, num_negatives: int = 1):
        with open(data_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[dict[str, str]]]:
        sample = self.samples[idx]
        query = sample['query']

        positives = sample["positive"]
        negatives = sample["negative"]
        if len(negatives) < self.num_negatives:
            negatives = negatives + [""] * (self.num_negatives - len(negatives))
        negatives = random.sample(negatives, self.num_negatives)
        docs = [positives] + negatives

        prompts = [PROMPT.format(query=query, document=doc) for doc in docs]
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        return messages


class DataCollatorForPointwise:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[list]) -> dict[str, torch.Tensor]:
        messages = [message for instance in instances for message in instance]
        model_inputs = self.tokenizer.apply_chat_template(
            messages, 
            padding=True, padding_side="left",
            max_length=256, truncation=True,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True
        )
        return dict(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
        )


class DistillationDataset(Dataset):

    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i) -> dict[str, Any]:
        documents = self.samples[i]["document"]
        messages = [
            [{"role": "user", "content": PROMPT.format(query=self.samples[i]["query"], document=documents[j])}] for j in range(len(documents))
        ]
        ranking = self.samples[i]["ranking"]

        return dict(
            messages=messages,
            ranking=ranking,
        )


class DataCollatorForDistillation:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[list]) -> dict[str, torch.Tensor]:
        messages, ranking = tuple([instance[key] for instance in instances] for key in ("messages", "ranking"))
        messages = [message for instance in messages for message in instance]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            padding=True, padding_side="left",
            max_length=256, truncation=True,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True
        )
        ranking = torch.tensor(ranking, dtype=torch.long) - 1  # make zero-indexed
        return dict(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            ranking=ranking,
        )


def make_data_module(tokenizer, data_args):
    if data_args.data_type == "pointwise":
         train_dataset = PointwiseDataset(data_path=data_args.data_path, num_negatives=data_args.num_negatives)
         eval_dataset = None
         if data_args.eval_data_path:
             eval_dataset = PointwiseDataset(data_path=data_args.eval_data_path, num_negatives=data_args.num_negatives)
         data_collator = DataCollatorForPointwise(tokenizer)
    elif data_args.data_type == "listwise":
        train_dataset = DistillationDataset(data_path=data_args.data_path)
        eval_dataset = None
        if data_args.eval_data_path:
            eval_dataset = DistillationDataset(data_path=data_args.eval_data_path)
        data_collator = DataCollatorForDistillation(tokenizer)
    else:
        raise ValueError(f"Unknown data_type: {data_args.data_type}")
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
