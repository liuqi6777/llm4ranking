import argparse
import collections
import datetime
import json
import os
import tempfile
from dataclasses import asdict
from tqdm import tqdm
from datasets import load_dataset

from llm4ranking import Reranker
from llm4ranking.evaluation.trec_eval import trec_eval, compute_metrics


def evaluate(
    rerank,
    datasets: list[str],
    retriever: str = "bm25",
    topk: int = 100,
    max_samples: int = None,
    output_dir: str = None,
):

    results = {}
    results["output_dir"] = output_dir

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        try:
            print(f"Evaluating dataset {dataset}...")
            
            data = load_dataset("liuqi6777/retrieval_results", data_files=f"{retriever}/{dataset}_top{topk}.jsonl", split="train").to_list()

            results[dataset] = {}
            if max_samples is not None:
                data = data[:max_samples]

            if dataset.startswith("bright"):
                task_name = dataset.removeprefix("bright-").replace("-", "_")
                examples = load_dataset('xlangai/bright', 'examples')[task_name]
                excluded_ids = {}
                for e in examples:
                    excluded_ids[e['id']] = e['excluded_ids']
            else:
                excluded_ids = None

            rerank_results = []
            records = []
            for i in tqdm(range(len(data))):
                _, rerank_indices, outputs = rerank(
                    query=data[i]["query"],
                    candidates=[x["content"] for x in data[i]["hits"]],
                    return_indices=True,
                    return_record=True
                )
                rerank_results.append({
                    "query": data[i]["query"],
                    "hits": [data[i]["hits"][j] for j in rerank_indices]
                })
                records.append(asdict(outputs))

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(
                    output_dir,
                    f"eval_{dataset}_top{topk}.txt"
                )
                with open(output_file, "w") as f:
                    write_results(rerank_results, f)
                records_file = os.path.join(
                    output_dir,
                    f"records_{dataset}_top{topk}.json"
                )
                metrics = trec_eval(dataset, output_file, excluded_ids)
                with open(records_file, "w") as f:
                    json.dump(records, f, indent=4, ensure_ascii=False)
            else:
                with tempfile.NamedTemporaryFile("w") as f:
                    write_results(rerank_results, f)
                    f.flush()
                    metrics = trec_eval(dataset, f.name, excluded_ids)

            results[dataset] = {}
            results[dataset]["metrics"] = metrics
        except Exception as e:
            raise e

    return results


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

    return evaluate(
        reranker.rerank,
        datasets=datasets,
        retriever=retriever,
        topk=topk,
        output_dir=output_dir
    )


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


def main(args):
    if args.output_dir is None:
        output_dir = os.path.join("results", "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        output_dir = args.output_dir
    if os.path.exists(os.path.join(output_dir, "results.json")) and not args.overwrite:
        print(f"Results exist in {output_dir}, pass...")
        return

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
        output_dir=output_dir
    )

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Results saved to {output_dir}")


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
    parser.add_argument("--return_record", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    main(args)
