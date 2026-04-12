import argparse
import ast
import collections
import datetime
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from statistics import mean
from tqdm import tqdm
from datasets import load_dataset

from llm4ranking import Reranker
from llm4ranking.evaluation.trec_eval import trec_eval, compute_metrics


def evaluate(
    rerank,
    datasets: list[str],
    retriever: str = "bm25",
    topk: int = 100,
    order: str = "initial",
    max_samples: int = None,
    output_dir: str = None,
    run_config: dict | None = None,
    reuse_predictions: bool = True,
):

    results = {}
    results["output_dir"] = output_dir

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        try:
            print(f"Evaluating dataset {dataset}...")
            
            data = load_dataset("liuqi6777/retrieval_results", data_files=f"{retriever}/{dataset}_top100.jsonl", split="train").to_list()

            results[dataset] = {}
            if max_samples is not None:
                data = data[:max_samples]

            if topk:
                if topk > 100:
                    print("Warning: only support top 100 now, will rerank top 100...")
                for i in range(len(data)):
                    data[i]["hits"] = data[i]["hits"][:topk]

            for i in range(len(data)):
                if order == "reverse":
                    data[i]["hits"].reverse()
                elif order == "random":
                    random.shuffle(data[i]["hits"])

            if dataset.startswith("bright"):
                task_name = dataset.removeprefix("bright-").replace("-", "_")
                examples = load_dataset('xlangai/bright', 'examples')[task_name]
                excluded_ids = {}
                for e in examples:
                    excluded_ids[e['id']] = e['excluded_ids']
            else:
                excluded_ids = None

            rerank_results = [None] * len(data)
            records = [None] * len(data)
            predictions_file = None
            resumed_count = 0
            config_signature = build_config_signature(run_config)

            if output_dir is not None:
                predictions_file = os.path.join(
                    output_dir,
                    f"predictions_{dataset}_top{topk}.jsonl"
                )
                if not reuse_predictions and os.path.exists(predictions_file):
                    os.remove(predictions_file)
                completed_predictions = {}
                if reuse_predictions:
                    completed_predictions = load_existing_predictions(
                        predictions_file=predictions_file,
                        data=data,
                        config_signature=config_signature,
                    )
                    for i, prediction in completed_predictions.items():
                        rerank_results[i] = {
                            "query": data[i]["query"],
                            "hits": [data[i]["hits"][j] for j in prediction["rerank_indices"]],
                        }
                        records[i] = prediction["record"]
                    resumed_count = len(completed_predictions)
            else:
                completed_predictions = {}

            pending_indices = [i for i in range(len(data)) if i not in completed_predictions]

            for i in tqdm(pending_indices):
                _, rerank_indices, outputs = rerank(
                    query=data[i]["query"],
                    candidates=[x["content"] for x in data[i]["hits"]],
                    return_indices=True,
                    return_record=True
                )
                record = asdict(outputs) if outputs else None
                rerank_result = {
                    "query": data[i]["query"],
                    "hits": [data[i]["hits"][j] for j in rerank_indices]
                }
                rerank_results[i] = rerank_result
                records[i] = record

                if predictions_file is not None:
                    append_prediction(
                        predictions_file=predictions_file,
                        entry=build_prediction_entry(
                            sample_idx=i,
                            sample=data[i],
                            rerank_indices=list(rerank_indices),
                            record=record,
                            config_signature=config_signature,
                        ),
                    )

            rerank_results = [item for item in rerank_results if item is not None]
            records = [item for item in records if item is not None]

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
                metrics_file = os.path.join(
                    output_dir,
                    f"metrics_{dataset}_top{topk}.json"
                )
                metrics = trec_eval(dataset, output_file, excluded_ids)
                with open(records_file, "w") as f:
                    json.dump(records, f, indent=4, ensure_ascii=False)
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4, ensure_ascii=False)
            else:
                with tempfile.NamedTemporaryFile("w") as f:
                    write_results(rerank_results, f)
                    f.flush()
                    metrics = trec_eval(dataset, f.name, excluded_ids)

            results[dataset] = {}
            results[dataset]["metrics"] = metrics
            results[dataset]["summary"] = build_summary(
                total_queries=len(data),
                completed_queries=len(rerank_results),
                resumed_queries=resumed_count,
                records=records,
                predictions_file=predictions_file,
            )
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
    order: str = "initial",
    reranking_args: dict = {},
    model_fw_args: dict = {},
    prompt_template: str = None,
    output_dir: str = None,
    reuse_predictions: bool = True,
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
        order=order,
        output_dir=output_dir,
        reuse_predictions=reuse_predictions,
        run_config={
            "model_type": model_type,
            "model_args": model_args,
            "datasets": datasets,
            "reranking_approach": reranking_approach,
            "retriever": retriever,
            "topk": topk,
            "order": order,
            "reranking_args": reranking_args,
            "model_fw_args": model_fw_args,
            "prompt_template": prompt_template,
        },
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


def build_config_signature(run_config: dict | None) -> str:
    payload = json.dumps(run_config or {}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prediction_signature(sample: dict) -> str:
    payload = {
        "query": sample["query"],
        "docids": [hit["docid"] for hit in sample["hits"]],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_prediction_entry(
    sample_idx: int,
    sample: dict,
    rerank_indices: list[int],
    record: dict | None,
    config_signature: str,
) -> dict:
    return {
        "sample_idx": sample_idx,
        "query": sample["query"],
        "qid": sample["hits"][0]["qid"] if sample["hits"] else None,
        "num_candidates": len(sample["hits"]),
        "rerank_indices": rerank_indices,
        "ranked_docids": [sample["hits"][idx]["docid"] for idx in rerank_indices],
        "prediction_signature": build_prediction_signature(sample),
        "config_signature": config_signature,
        "record": record,
    }


def append_prediction(predictions_file: str, entry: dict):
    with open(predictions_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False))
        f.write("\n")


def load_existing_predictions(
    predictions_file: str,
    data: list[dict],
    config_signature: str,
) -> dict[int, dict]:
    if not os.path.exists(predictions_file):
        return {}

    completed_predictions = {}
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            prediction = json.loads(line)
            sample_idx = prediction["sample_idx"]
            if sample_idx >= len(data):
                continue
            if prediction.get("config_signature") != config_signature:
                continue
            if prediction.get("prediction_signature") != build_prediction_signature(data[sample_idx]):
                continue
            completed_predictions[sample_idx] = prediction
    return completed_predictions


def build_summary(
    total_queries: int,
    completed_queries: int,
    resumed_queries: int,
    records: list[dict | None],
    predictions_file: str | None,
) -> dict:
    latencies = [
        record["latency"] for record in records
        if record is not None and record.get("latency") is not None
    ]
    return {
        "total_queries": total_queries,
        "completed_queries": completed_queries,
        "resumed_queries": resumed_queries,
        "newly_computed_queries": completed_queries - resumed_queries,
        "avg_latency": round(mean(latencies), 4) if latencies else None,
        "total_latency": round(sum(latencies), 4) if latencies else None,
        "predictions_file": predictions_file,
    }


def parse_dict_args(args_string: str):
    args = {}
    for arg in args_string.split(","):
        key, value = arg.strip().split("=")
        try:
            args[key] = ast.literal_eval(value)
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
        order=args.order,
        reranking_args=args.reranking_args,
        model_fw_args=args.model_fw_args,
        prompt_template=args.prompt_template,
        output_dir=output_dir,
        reuse_predictions=not args.overwrite,
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
    parser.add_argument("--order", type=str, default="initial", choices=["initial", "random", "reverse"])
    parser.add_argument("--reranking_args", type=parse_dict_args, default={})
    parser.add_argument("--model_fw_args", type=parse_dict_args, default={})
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--return_record", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    main(args)
