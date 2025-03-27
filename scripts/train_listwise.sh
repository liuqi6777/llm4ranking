#!/bin/bash

torchrun src/llm4ranking/training/listwise/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --data_path ./toy_data/listwise_train.jsonl \
    --bf16 True \
    --tf32 True \
    --output_dir "path/to/your/checkpoints" \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
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
