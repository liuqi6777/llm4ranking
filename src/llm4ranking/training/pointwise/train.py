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


def get_model(model_args, lora_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
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

    model = get_model(model_args, lora_args)
    model.train()

    data_module = make_data_module(tokenizer, data_args)
    train_dataset = data_module['train_dataset']
    data_collator = data_module['data_collator']

    class PointwiseTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            logits = model(**inputs).logits
            scores = logits[:, -1, yes_loc].view(self.args.per_device_train_batch_size, -1)
            targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
            loss = torch.nn.functional.cross_entropy(scores, targets, reduction='mean')
            return loss

    trainer = PointwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save model
    if training_args.output_dir:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
