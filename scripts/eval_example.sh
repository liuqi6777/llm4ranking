#!/bin/bash

set -e

model=meta-llama/Llama-3.1-8B-Instruct
model=Qwen/Qwen2.5-7B-Instruct


python -m llm4ranking.evaluation.evaluator \
  --model_type hf \
  --model_args model=$model \
  --reranking_approach rankgpt \
  --reranking_args window_size=20,step=10,truncate_length=300 \
  --datasets dl19 \
  --model_fw_args do_sample=False,max_new_tokens=128 \
  --topk 20


python -m llm4ranking.evaluation.evaluator \
  --model_type hf \
  --model_args model=$model \
  --reranking_approach rel-gen \
  --datasets dl19 \
  --topk 20

