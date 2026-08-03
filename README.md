# LLM4Ranking: An Easy-to-use Framework of Utilizing Large Language Models for Document Reranking
Large language models for ranking.

## Installation

LLM4Ranking can be easily installed from source via the following methods:

```bash
git clone git@github.com:liuqi6777/llm4ranking.git
cd llm4ranking
pip install -e .
```

Install the optional vLLM backend with:

```bash
pip install -e ".[vllm]"
```

## Minimal Usage Example

To illustrate the fundamental functionality of our framework, we provide a minimal usage example that can rerank documents using just a few lines of code:
```python
from llm4ranking import BackendRuntimeArgs, Reranker, StrategyRuntimeArgs

reranker = Reranker(
    reranking_approach="rankgpt",
    model_type="openai",
    model_name="gpt-4o",
)

result = reranker.rerank(
    query="query text",
    candidates=["doc0", "doc1", "doc2"],
    strategy=StrategyRuntimeArgs(
        return_record=True,
        window_size=20,
        step=10,
    ),
    backend=BackendRuntimeArgs(
        max_completion_tokens=128,
        temperature=0,
    ),
)

print(result.documents)
print(result.indices)
```

### Supported LLMs

The framework supports Hugging Face Transformers, vLLM, OpenAI, and other APIs compatible with the OpenAI API. The vLLM backend supports batched generation, assistant-response log likelihood, and batched next-token scoring:

```python
reranker = Reranker(
    reranking_approach="rel-gen",
    model_type="vllm",
    model_name="Qwen/Qwen2.5-7B-Instruct",
    model_args={
        "tensor_parallel_size": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

The `logits` interface returns vLLM next-token log probabilities. They are equivalent to raw logits for softmax and ranking because they differ by a token-independent normalization constant. Full-vocabulary scoring, as used by FIRST, can consume substantially more host memory than scoring a short label-token list.

### Supported Reranking Models

You can list all the supported reranking models by running the following command:
```bash
python -m llm4ranking.list_reranking_models
```
or running the following code:
```python
from llm4ranking import list_available_reranking_approaches

list_available_reranking_approaches()
```
More details are coming soon. You can refer to [Awesome-LLM4Ranking](https://github.com/liuqi6777/Awesome-LLM4Ranking) for more information.

## Train Your Own Model

We provide training script example in `scripts`.

## Evaluation on Benchmarks

To evaluate an reranking model, you can run the following command:
```bash
model=Qwen/Qwen2.5-7B-Instruct

python -m llm4ranking.evaluation.evaluator \
  --model_type hf \
  --model_args model='$model' \
  --reranking_approach rankgpt \
  --strategy_args window_size=20,step=10,truncate_length=300 \
  --datasets dl19 \
  --backend_args do_sample=False,max_new_tokens=128 \
  --topk 20
```

### Supported Datasets

The framework supports the evaluation of the following datasets:
- TREC DL 2019 and 2020
- BEIR
- MAIR
- BRIGHT

More examples can be found in the `scripts` directory.

## Citation

If you found this repository helpful, please cite the following paper:

```
@misc{liu2025llm4rankingeasytouseframeworkutilizing,
      title={LLM4Ranking: An Easy-to-use Framework of Utilizing Large Language Models for Document Reranking}, 
      author={Qi Liu and Haozhe Duan and Yiqun Chen and Quanfeng Lu and Weiwei Sun and Jiaxin Mao},
      year={2025},
      eprint={2504.07439},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2504.07439}, 
}
```
