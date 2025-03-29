import ujson as json
from typing import Dict, List, Sequence
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        query = sample['query']

        def get_text(x):
            if x["title"]:
                return f"{x['title']} {x['text']}"
            return x["text"]

        positives = [get_text(x) for x in sample["positive_passages"]]
        positives = random.sample(positives, 1)
        negatives = [get_text(x) for x in sample["negative_passages"]]
        if len(negatives) < self.num_negatives:
            negatives = negatives + [""] * (self.num_negatives - len(negatives))
        negatives = random.sample(negatives, self.num_negatives)
        docs = positives + negatives
        prompts = [PROMPT.format(query=query, document=doc) for doc in docs]
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        return messages


class DataCollatorForPointwise:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[list]) -> Dict[str, torch.Tensor]:
        messages = [message for instance in instances for message in instance]
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            padding=True, padding_side="left",
            max_length=256, truncation=True,
            return_tensors='pt',
            add_generation_prompt=True,
            return_dict=True
        )
        return inputs


def make_data_module(tokenizer, data_args):
    train_dataset = PointwiseDataset(data_path=data_args.data_path, num_negatives=data_args.num_negatives)
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = PointwiseDataset(data_path=data_args.eval_data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForPointwise(tokenizer)
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator
    }
