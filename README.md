# LLM4Ranking: An Easy-to-use Framework of Utilizing Large Language Models for Document Reranking
Large language models for ranking.

## Installation

LLM4Ranking can be easily installed from source via the following methods:

```bash
git clone git@github.com:liuqi6777/llm4ranking.git
cd llm4ranking
pip install -e .
```

## Minimal Usage Example

To illustrate the fundamental functionality of our framework, we provide a minimal usage example that can rerank documents using just a few lines of code:
```python
from llm4ranking import Reranker

reranker = Reranker(
    reranking_approach="rankgpt",
    model_type="openai", model_name="gpt-4o"
)
reranker.rerank(
    query: "query text",
    candidates: ["doc0", "doc1", "doc2", ...],
)
```

### Supported LLMs

The framework supports both open-source and commercial LLMs. For open-source LLMs, we support Hugging Face Transformers. For commercial LLMs, we support OpenAI and other LLMs that are compatible with the OpenAI API.

### Supported Reranking Models

You can list all the supported reranking models by running the following command:
```bash
python -m llm4ranking.list_reranking_models
```
or running the following code:
```python
from llm4ranking import list_reranking_models

list_reranking_models()
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
  --model_args model=$model \
  --reranking_approach rankgpt \
  --reranking_args window_size=20,step=10,truncate_length=300 \
  --datasets dl19 \
  --model_fw_args do_sample=False,max_new_tokens=128 \
  --topk 20
```

### Supported Datasets

Coming soon.

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
