from dataclasses import dataclass
import ujson as json
from typing import Sequence, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset
import transformers


IGNORE_TOKEN_ID = -100


def preprocess_messages(
    tokenizer: transformers.PreTrainedTokenizer,
    messages: list[dict[str, str]],
) -> dict[str, Tensor]:
    if messages[-1]["role"] == "assistant":
        if messages[-1]["content"].startswith("["):
            messages[-1]["content"] = ' ' + messages[-1]["content"]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    targets = input_ids.clone()
    targets = _mask_targets(tokenizer, messages, targets)
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def _get_messages_length(
    messages: list[dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        return_tensors='pt',
        max_length=tokenizer.model_max_length,
        truncation=True
    ).shape[1]


def _mask_targets(
    tokenizer: transformers.PreTrainedTokenizer,
    messages: list[dict[str, str]],
    targets: Tensor,
) -> Tensor:
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            message_start_idx = _get_messages_length(messages[:message_idx], tokenizer) if message_idx > 0 else 0
            message_end_idx = _get_messages_length(messages[:message_idx+1], tokenizer)         
            targets[:, message_start_idx:message_end_idx] = IGNORE_TOKEN_ID
            if message_end_idx >= tokenizer.model_max_length:
                break
    return targets


class SFTDatasetForCausalLM(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            self.raw_data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.raw_data)} examples from {data_path}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, Tensor]:
        ret = preprocess_messages(
            self.tokenizer,
            self.raw_data[i]["messages"],
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )

        return ret


@dataclass
class DataCollatorForCausalLM:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> dict:
    train_dataset = SFTDatasetForCausalLM(
        data_args.data_path,
        tokenizer=tokenizer,
    )
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
