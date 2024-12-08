# llm4ranking
Large language models for ranking.

## Installation

```bash
git clone git@github.com:liuqi6777/llm4ranking.git
cd llm4ranking
pip install -e .
```


## Evaluation Example

```bash
python -m llm4ranking.evaluation.evaluator \
  --model_type hf \
  --model_args '{"model": "meta-llama/Llama-3.1-8B-Instruct"}' \
  --reranking_approach listwise-sw \
  --datasets dl19 \
  --model_fw_args '{"do_sample": false}'
```
