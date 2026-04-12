import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from llm4ranking.training.pointwise.arguments import (
    DataArguments,
    LoraArguments,
    ModelArguments,
    TrainingArguments
)
from llm4ranking.training.pointwise.data import make_data_module
from llm4ranking.training.pointwise.loss import rank_net


def get_model(model_args, lora_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    model.use_cache = False
    if lora_args.lora_enabled:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


def main():
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,  # Use user's TrainingArguments
        LoraArguments
    ))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    yes_loc = tokenizer.encode("yes")[0]
    no_loc = tokenizer.encode("no")[0]

    model = get_model(model_args, lora_args)
    model.train()

    data_module = make_data_module(tokenizer, data_args)
    num_docs_per_query = data_args.num_negatives + 1

    class PointwiseTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            logits = outputs.logits[:, -1, :]
            if "ranking" not in inputs:
                if logits.shape[0] % num_docs_per_query != 0:
                    raise ValueError(
                        f"Expected batch size to be divisible by {num_docs_per_query}, got {logits.shape[0]}."
                    )
                batch_size = logits.shape[0] // num_docs_per_query
                scores = torch.softmax(logits[:, [yes_loc, no_loc]], dim=-1)[:, 0].view(
                    batch_size, num_docs_per_query
                )
                targets = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
                loss = torch.nn.functional.cross_entropy(scores / 0.1, targets, reduction='mean')
            else:
                batch_size, slate_length = inputs["ranking"].shape
                rank_position = torch.empty_like(inputs["ranking"], device=inputs["ranking"].device, dtype=torch.long)
                rank_indices = torch.arange(slate_length, device=inputs["ranking"].device).expand(batch_size, -1)
                rank_position.scatter_(dim=1, index=inputs["ranking"], src=rank_indices)
                if self.args.loss_type == "ce":
                    scores = torch.softmax(logits[:, [yes_loc, no_loc]], dim=-1)[:, 0].view(batch_size, slate_length)
                    targets = rank_position[:, 0]
                    loss = torch.nn.functional.cross_entropy(scores / 0.1, targets, reduction='mean')
                elif self.args.loss_type == "ranknet":
                    scores = logits[:, yes_loc].view(batch_size, slate_length)
                    loss = rank_net(scores, slate_length - rank_position)
            return (loss, outputs) if return_outputs else loss

    trainer = PointwiseTrainer(
        model=model,
        args=training_args,
        **data_module,
    )

    # Train
    trainer.train()
    torch.cuda.synchronize()
    trainer.save_model()


if __name__ == "__main__":
    main()
