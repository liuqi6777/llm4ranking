import argparse
import datetime
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
                    f"eval_{dataset}_top{topk}_pass{pass_}.txt"
                )
                with open(output_file, "w") as f:
                    write_results(rerank_results, f)
                metrics = trec_eval(dataset, output_file)
            else:
                with tempfile.NamedTemporaryFile("w") as f:
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


def parse_dict_args(args_string: str):
    args = {}
    for arg in args_string.split(","):
        key, value = arg.strip().split("=")
        try:
            args[key] = eval(value)
        except Exception:
            args[key] = value
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_args", type=parse_dict_args, required=True)
    parser.add_argument("--reranking_approach", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--reranking_args", type=parse_dict_args, default={})
    parser.add_argument("--model_fw_args", type=parse_dict_args, default={})
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--num_passes", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    print(args)

    if args.output_dir is None:
        output_dir = os.path.join("results", "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = args.output_dir

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    results = simple_evaluate(
        model_type=args.model_type,
        model_args=args.model_args,
        datasets=args.datasets,
        reranking_approach=args.reranking_approach,
        retriever=args.retriever,
        topk=args.topk,
        reranking_args=args.reranking_args,
        model_fw_args=args.model_fw_args,
        prompt_template=args.prompt_template,
        num_passes=args.num_passes,
        output_dir=output_dir
    )

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_dir}")
