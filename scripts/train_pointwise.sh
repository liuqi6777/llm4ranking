#!/bin/bash

torchrun src/llm4ranking/training/logits/train.py \
    --deepspeed scripts/ds_zero3.json \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --data_path ./data/pointwise_train.jsonl \
    --bf16 True \
    --tf32 True \
    --output_dir "path/to/your/checkpoints" \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --max_steps 1000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_negatives 3 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True
