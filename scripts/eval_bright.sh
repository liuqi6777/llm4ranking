#!/bin/bash

set -e

model=Qwen/Qwen2.5-7B-Instruct


python -m llm4ranking.evaluation.evaluate_bright \
  --model_type hf \
  --model_args model=$model \
  --reranking_approach rankgpt \
  --reranking_args window_size=20,step=10,truncate_length=300 \
  --tasks pony \
  --model_fw_args do_sample=False,max_new_tokens=128 \
  --topk 20

