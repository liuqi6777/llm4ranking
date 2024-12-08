import json
import os
import tempfile
from functools import partial
from tqdm import tqdm
from datasets import load_dataset

from llm4ranking.ranker import *
from llm4ranking.model import *
from llm4ranking.evaluation.trec_eval import trec_eval


RERANKING_APPROACHES = {
    "listwise-sw": (ListwiseSilidingWindowReranker, ListwiseGeneration),
    "pointwise-rg": (PointwiseReranker, RelevanceGeneration),
    "pairwise": (PairwiseReranker, PairwiseComparison),
}


def simple_rerank(
    query: str,
    candidates: list[str],
    reranking_approach: str,
    model_type: str,
    model_args: dict,
    reranking_args: dict = {},
    model_fw_args: dict = {},
    prompt_template: str = None,
):
    reranker = RERANKING_APPROACHES[reranking_approach][0]()
    ranking_func = RERANKING_APPROACHES[reranking_approach][1](model_type, model_args, prompt_template)
    return reranker.rerank(query, candidates, ranking_func, **reranking_args, **model_fw_args)


def simple_evaluate(
    model_type: str,
    model_args: dict,
    datasets: list[str],
    reranking_approach: str,
    retriever: str = "bm25",
    topk: int = 100,
    reranking_args: dict = {},
    model_fw_args: dict = {},
    prompt_template: str = None,
    num_passes: int = 1,
    output_dir: str = None,
):
    reranker = RERANKING_APPROACHES[reranking_approach][0]()
    ranking_func = RERANKING_APPROACHES[reranking_approach][1](model_type, model_args, prompt_template)
    rerank = partial(reranker.rerank, ranking_func=ranking_func, **reranking_args, **model_fw_args)

    results = {}

    results["args"] = {
        "model_type": model_type,
        "model_args": model_args,
        "datasets": datasets,
        "reranking_approach": reranking_approach,
        "retriever": retriever,
        "topk": topk,
        "reranking_args": reranking_args,
        "model_fw_args": model_fw_args,
        "prompt_template": prompt_template,
        "num_passes": num_passes,
    }
    results["metrics"] = {}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        data = load_dataset("liuqi6777/pyserini_retrieval_results", data_files=f"{retriever}/{dataset}_top{topk}.jsonl", split="train")
        results["metrics"][dataset] = {}

        prev_results = data
        for pass_ in range(num_passes):
            rerank_results = []
            for i in tqdm(range(len(prev_results))):
                _, rerank_indices = rerank(
                    query=prev_results[i]["query"],
                    candidates=[x["content"] for x in prev_results[i]["hits"]]
                )
                rerank_results.append({
                    "query": prev_results[i]["query"],
                    "hits": [prev_results[i]["hits"][j] for j in rerank_indices]
                })
            prev_results = rerank_results

            if output_dir is not None:
                output_file = os.path.join(
                    output_dir,
                    f"eval_{dataset}_{ranking_func.model_name.split('/')[-1]}_top{topk}_pass{pass_}.txt"
                )
                with open(output_file, "w") as f:
                    write_results(rerank_results, f)
                    metrics = trec_eval(dataset, f.name)
            else:
                with tempfile.NamedTemporaryFile("w", delete=False) as f:
                    write_results(rerank_results, f)
                    metrics = trec_eval(dataset, f.name)

            results["metrics"][dataset]["pass" + str(pass_)] = metrics

    return results


def write_results(rerank_results, file_obj):
    for i, item in enumerate(rerank_results):
        hits = item["hits"]
        for j, hit in enumerate(hits):
            file_obj.write(f"{hit['qid']} Q{i} {hit['docid']} {j + 1} {round(1 / (j + 1), 3)} rank")
            file_obj.write("\n")


if __name__ == "__main__":
    simple_evaluate(
        model_type="hf",
        model_args={"model": "meta-llama/Llama-3.1-8B-Instruct"},
        reranking_approach="listwise-sw",
        datasets=["dl19"],
        retriever="bm25",
        topk=20,
        model_fw_args={"do_sample": False},
    )
