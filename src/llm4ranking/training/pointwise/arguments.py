from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="gpt-3.5-turbo",
        metadata={"help": "LLM model to use"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code for the model"}
    )
    padding_side: str = field(
        default="left",
        metadata={"help": "Padding side for tokenizer"}
    )


@dataclass
class DataArguments:
    data_type: str = field(
        default="pointwise",
        metadata={"help": "Type of data: pointwise or distillation"}
    )
    data_path: str = field(
        default=None,
        metadata={"help": "Path to training data file", "required": True}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data file"}
    )
    num_negatives: int = field(
        default=1,
        metadata={"help": "Number of negative samples per positive sample"}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    loss_type: str = field(
        default="ce",
        metadata={"help": "Type of loss function to use"}
    )
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class LoraArguments:
    lora_enabled: bool = field(
        default=False,
        metadata={"help": "Enable LoRA training"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA adaptation"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias configuration"}
    )
