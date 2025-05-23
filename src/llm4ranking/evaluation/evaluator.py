import argparse
import collections
import datetime
import json
import os
import tempfile
from dataclasses import asdict
from functools import partial
from tqdm import tqdm
from datasets import load_dataset

from llm4ranking import Reranker
from llm4ranking.evaluation.trec_eval import trec_eval, compute_metrics


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
    reranker = Reranker(
        reranking_approach=reranking_approach,
        model_type=model_type,
        model_name=model_args["model"],
        model_args=model_args,
        prompt_template=prompt_template,
        reranking_args=reranking_args,
        model_fw_args=model_fw_args,
    )

    results = {}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        data = load_dataset("liuqi6777/pyserini_retrieval_results", data_files=f"{retriever}/{dataset}_top{topk}.jsonl", split="train")
        results[dataset] = {}

        prev_results = data
        for pass_ in range(num_passes):
            rerank_results = []
            all_records = []
            for i in tqdm(range(len(prev_results))):
                _, rerank_indices, record = reranker.rerank(
                    query=prev_results[i]["query"],
                    candidates=[x["content"] for x in prev_results[i]["hits"]],
                    return_record=True,
                    return_indices=True
                )
                rerank_results.append({
                    "query": prev_results[i]["query"],
                    "hits": [prev_results[i]["hits"][j] for j in rerank_indices]
                })
                all_records.append(asdict(record))
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

            results[dataset]["pass" + str(pass_)] = {}
            results[dataset]["pass" + str(pass_)]["metrics"] = metrics
            results[dataset]["pass" + str(pass_)]["records"] = all_records

    return results


def evaluate_one_dataset(
    queries,
    query_ids,
    documents,
    doc_ids,
    qrels,
    reranker,
):
    run = collections.defaultdict(dict)
    for query, query_id, one_docs, one_doc_ids in tqdm(zip(queries, query_ids, documents, doc_ids)):
        _, rerank_indices, *_ = reranker.rerank(
            query=query,
            candidates=one_docs,
            return_indices=True
        )
        for rank, indice in enumerate(rerank_indices):
            run[query_id][one_doc_ids[indice]] = round(1 / (rank + 1), 3)
    metrics = compute_metrics(qrels, run)
    return metrics


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
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cli_args.json"), "w") as f:
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
        json.dump(results, f, indent=4, default=str)
    print(f"Results saved to {output_dir}")
