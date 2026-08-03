import argparse
import ast
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from tqdm import tqdm

from llm4ranking import ModelConfig, Reranker, RerankerConfig


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
    seed: int = 42,
    retrieval_revision: str | None = None,
    dataset_revision: str | None = None,
):
    from datasets import load_dataset
    from llm4ranking.evaluation.trec_eval import trec_eval

    effective_run_config = dict(run_config or {})
    effective_run_config["seed"] = seed
    effective_run_config.setdefault("retrieval_revision", retrieval_revision)
    effective_run_config.setdefault("dataset_revision", dataset_revision)
    results = {}
    results["output_dir"] = output_dir
    results["seed"] = seed

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        write_run_metadata(output_dir, run_config=effective_run_config, seed=seed)

    for dataset in datasets:
        try:
            print(f"Evaluating dataset {dataset}...")

            data = load_dataset(
                "liuqi6777/retrieval_results",
                data_files=f"{retriever}/{dataset}_top100.jsonl",
                split="train",
                revision=retrieval_revision,
            ).to_list()

            results[dataset] = {}
            if max_samples is not None:
                data = data[:max_samples]

            if topk:
                if topk > 100:
                    print("Warning: only support top 100 now, will rerank top 100...")
                for i in range(len(data)):
                    data[i]["hits"] = data[i]["hits"][:topk]

            order_rng = random.Random(derive_seed(seed, dataset, "candidate-order"))
            for i in range(len(data)):
                if order == "reverse":
                    data[i]["hits"].reverse()
                elif order == "random":
                    order_rng.shuffle(data[i]["hits"])

            if dataset.startswith("bright"):
                task_name = dataset.removeprefix("bright-").replace("-", "_")
                examples = load_dataset(
                    "xlangai/bright",
                    "examples",
                    revision=dataset_revision,
                )[task_name]
                excluded_ids = {}
                for e in examples:
                    excluded_ids[e["id"]] = e["excluded_ids"]
            else:
                excluded_ids = None

            rerank_results = [None] * len(data)
            records = [None] * len(data)
            predictions_file = None
            resumed_count = 0
            config_signature = build_config_signature(effective_run_config)

            if output_dir is not None:
                predictions_file = os.path.join(
                    output_dir, f"predictions_{dataset}_top{topk}.jsonl"
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
                            "hits": [
                                data[i]["hits"][j] for j in prediction["rerank_indices"]
                            ],
                        }
                        records[i] = prediction["record"]
                    resumed_count = len(completed_predictions)
            else:
                completed_predictions = {}

            pending_indices = [
                i for i in range(len(data)) if i not in completed_predictions
            ]

            for i in tqdm(pending_indices):
                query_seed = derive_seed(seed, dataset, i)
                seed_everything(query_seed)
                result = rerank(
                    query=data[i]["query"],
                    candidates=[x["content"] for x in data[i]["hits"]],
                    return_record=True,
                )
                record = asdict(result.record) if result.record else None
                if record is not None:
                    record["seed"] = query_seed
                rerank_result = {
                    "query": data[i]["query"],
                    "hits": [data[i]["hits"][j] for j in result.indices],
                }
                rerank_results[i] = rerank_result
                records[i] = record

                if predictions_file is not None:
                    append_prediction(
                        predictions_file=predictions_file,
                        entry=build_prediction_entry(
                            sample_idx=i,
                            sample=data[i],
                            rerank_indices=list(result.indices),
                            record=record,
                            config_signature=config_signature,
                        ),
                    )

            rerank_results = [item for item in rerank_results if item is not None]
            records = [item for item in records if item is not None]

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"eval_{dataset}_top{topk}.txt")
                with open(output_file, "w") as f:
                    write_results(rerank_results, f)
                records_file = os.path.join(
                    output_dir, f"records_{dataset}_top{topk}.json"
                )
                metrics_file = os.path.join(
                    output_dir, f"metrics_{dataset}_top{topk}.json"
                )
                metrics = trec_eval(dataset, output_file, excluded_ids)
                with open(records_file, "w") as f:
                    json.dump(
                        records,
                        f,
                        indent=4,
                        ensure_ascii=False,
                        default=json_default,
                    )
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
    strategy_args: dict | None = None,
    backend_args: dict | None = None,
    prompt_template: str = None,
    output_dir: str = None,
    reuse_predictions: bool = True,
    seed: int = 42,
    retrieval_revision: str | None = None,
    dataset_revision: str | None = None,
):
    reranker = Reranker(
        reranking_approach=reranking_approach,
        model_type=model_type,
        model_name=model_args["model"],
        model_args=model_args,
        prompt_template=prompt_template,
        strategy_args=strategy_args,
        backend_args=backend_args,
    )

    return evaluate(
        reranker.rerank,
        datasets=datasets,
        retriever=retriever,
        topk=topk,
        order=order,
        output_dir=output_dir,
        reuse_predictions=reuse_predictions,
        seed=seed,
        retrieval_revision=retrieval_revision,
        dataset_revision=dataset_revision,
        run_config={
            "model_type": model_type,
            "model_args": model_args,
            "datasets": datasets,
            "reranking_approach": reranking_approach,
            "retriever": retriever,
            "topk": topk,
            "order": order,
            "strategy_args": strategy_args,
            "backend_args": backend_args,
            "prompt_template": prompt_template,
            "seed": seed,
            "retrieval_revision": retrieval_revision,
            "dataset_revision": dataset_revision,
        },
    )


def evaluate_one_dataset(
    queries,
    query_ids,
    documents,
    doc_ids,
    qrels,
    reranker,
    seed: int = 42,
    dataset_name: str = "dataset",
):
    from llm4ranking.evaluation.trec_eval import compute_metrics

    run = collections.defaultdict(dict)
    samples = zip(queries, query_ids, documents, doc_ids)
    for sample_idx, (query, query_id, one_docs, one_doc_ids) in enumerate(
        tqdm(samples)
    ):
        seed_everything(derive_seed(seed, dataset_name, sample_idx))
        result = reranker.rerank(query=query, candidates=one_docs)
        for rank, indice in enumerate(result.indices):
            run[query_id][one_doc_ids[indice]] = len(result.indices) - rank
    metrics = compute_metrics(qrels, run)
    return metrics


def write_results(rerank_results, file_obj):
    for i, item in enumerate(rerank_results):
        hits = item["hits"]
        for j, hit in enumerate(hits):
            score = len(hits) - j
            file_obj.write(f"{hit['qid']} Q{i} {hit['docid']} {j + 1} {score} rank")
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
        f.write(json.dumps(entry, ensure_ascii=False, default=json_default))
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
            if prediction.get("prediction_signature") != build_prediction_signature(
                data[sample_idx]
            ):
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
        record["latency"]
        for record in records
        if record is not None and record.get("latency") is not None
    ]
    valid_records = [record for record in records if record is not None]
    peak_memory = [
        record["peak_memory_mb"]
        for record in valid_records
        if record.get("peak_memory_mb") is not None
    ]
    return {
        "total_queries": total_queries,
        "completed_queries": completed_queries,
        "resumed_queries": resumed_queries,
        "newly_computed_queries": completed_queries - resumed_queries,
        "avg_latency": round(mean(latencies), 4) if latencies else None,
        "total_latency": round(sum(latencies), 4) if latencies else None,
        "total_lm_calls": sum(record.get("num_lm_calls", 0) for record in valid_records),
        "total_input_tokens": sum(record.get("num_input_tokens", 0) for record in valid_records),
        "total_output_tokens": sum(record.get("num_output_tokens", 0) for record in valid_records),
        "total_parse_failures": sum(record.get("num_parse_failures", 0) for record in valid_records),
        "total_fallbacks": sum(record.get("num_fallbacks", 0) for record in valid_records),
        "total_truncated_docs": sum(record.get("num_truncated_docs", 0) for record in valid_records),
        "peak_memory_mb": max(peak_memory) if peak_memory else None,
        "predictions_file": predictions_file,
    }


def derive_seed(base_seed: int, dataset: str, sample: int | str) -> int:
    payload = f"{base_seed}:{dataset}:{sample}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def json_default(value):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "tolist"):
        size = int(getattr(value, "size", 0))
        if size <= 256:
            return value.tolist()
        raw = value.tobytes() if hasattr(value, "tobytes") else repr(value.tolist()).encode("utf-8")
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "values_omitted": True,
        }
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_reproducibility_metadata(seed: int) -> dict:
    packages = {}
    for package in ("llm4ranking", "torch", "transformers", "datasets", "vllm", "openai"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    metadata = {
        "seed": seed,
        "per_query_seed": "sha256(base_seed:dataset:sample_index)[0:4]",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source": get_source_metadata(),
    }
    try:
        import torch

        metadata["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except ImportError:
        metadata["torch"] = None
    return metadata


def get_source_metadata() -> dict | None:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        return {"git_commit": commit, "git_dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_metadata(output_dir: str, run_config: dict | None, seed: int) -> None:
    metadata = {
        "run_config": run_config or {},
        "reproducibility": build_reproducibility_metadata(seed),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False, default=json_default)


def parse_dict_args(args_string: str):
    if args_string is None:
        return {}
    if isinstance(args_string, dict):
        return dict(args_string)
    if args_string.strip() == "":
        return {}
    args = {}
    for arg in args_string.split(","):
        key, value = arg.strip().split("=", 1)
        try:
            args[key] = ast.literal_eval(value)
        except Exception:
            args[key] = value
    return args


def add_reranker_cli_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="openai")
    parser.add_argument("--model_args", type=parse_dict_args, default=None)
    parser.add_argument("--reranking_approach", type=str, default="rankgpt")
    parser.add_argument("--strategy_args", type=parse_dict_args, default=None)
    parser.add_argument("--backend_args", type=parse_dict_args, default=None)
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval_revision", type=str, default=None)
    parser.add_argument("--dataset_revision", type=str, default=None)
    return parser


def reranker_config_from_mapping(data: dict) -> RerankerConfig:
    model_data = dict(data.get("model", {}))
    return RerankerConfig(
        reranking_approach=data.get("reranking_approach", "rankgpt"),
        model=ModelConfig(
            model_type=model_data.get("model_type", "openai"),
            model_name=model_data.get("model_name"),
            init_args=dict(model_data.get("init_args", {})),
            inference_args=dict(model_data.get("inference_args", {})),
        ),
        strategy_args=dict(data.get("strategy_args", {})),
        prompt_template=data.get("prompt_template"),
    )


def build_reranker_from_cli_args(args) -> Reranker:
    if args.config_json:
        config = reranker_config_from_mapping(json.loads(args.config_json))
        return Reranker.from_config(config)

    return Reranker(
        reranking_approach=args.reranking_approach,
        model_type=args.model_type,
        model_name=(args.model_args or {}).get("model"),
        model_args=args.model_args,
        strategy_args=args.strategy_args,
        backend_args=args.backend_args,
        prompt_template=args.prompt_template,
    )


def build_run_config_from_cli_args(args, datasets: list[str]) -> dict:
    if args.config_json:
        return {
            "config": json.loads(args.config_json),
            "datasets": datasets,
            "retriever": args.retriever,
            "topk": args.topk,
            "order": args.order,
            "seed": args.seed,
            "retrieval_revision": args.retrieval_revision,
            "dataset_revision": args.dataset_revision,
        }

    return {
        "model_type": args.model_type,
        "model_args": args.model_args,
        "datasets": datasets,
        "reranking_approach": args.reranking_approach,
        "retriever": args.retriever,
        "topk": args.topk,
        "order": args.order,
        "strategy_args": args.strategy_args,
        "backend_args": args.backend_args,
        "prompt_template": args.prompt_template,
        "seed": args.seed,
        "retrieval_revision": args.retrieval_revision,
        "dataset_revision": args.dataset_revision,
    }


def main(args):
    if args.output_dir is None:
        output_dir = os.path.join(
            "results", "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    else:
        output_dir = args.output_dir
    if os.path.exists(os.path.join(output_dir, "results.json")) and not args.overwrite:
        print(f"Results exist in {output_dir}, pass...")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cli_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    reranker = build_reranker_from_cli_args(args)
    run_config = build_run_config_from_cli_args(args, datasets=args.datasets)
    results = evaluate(
        reranker.rerank,
        datasets=args.datasets,
        retriever=args.retriever,
        topk=args.topk,
        order=args.order,
        output_dir=output_dir,
        reuse_predictions=not args.overwrite,
        run_config=run_config,
        seed=args.seed,
        retrieval_revision=args.retrieval_revision,
        dataset_revision=args.dataset_revision,
    )

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_reranker_cli_arguments(parser)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--order", type=str, default="initial", choices=["initial", "random", "reverse"]
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    if not args.config_json and args.model_args is None:
        parser.error("Either --config_json or --model_args must be provided.")

    main(args)
