import argparse
import datetime
import json
import os

from llm4ranking.evaluation.evaluator import (
    add_reranker_cli_arguments,
    build_reranker_from_cli_args,
    evaluate_one_dataset,
)
from llm4ranking.evaluation.utils import load_bright, retrieval_bm25
from llm4ranking.evaluation.trec_eval import trec_eval, compute_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_reranker_cli_arguments(parser)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    print(args)

    if not args.config_json and args.model_args is None:
        parser.error("Either --config_json or --model_args must be provided.")

    if args.output_dir is None:
        output_dir = os.path.join(
            "results", "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cli_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    reranker = build_reranker_from_cli_args(args)

    for task in args.tasks:
        print(f"Evaluating task: {task}")
        all_results = {}
        dataset = load_bright(task)
        if args.retriever == "bm25":
            retrieval_results = retrieval_bm25(
                queries=dataset["queries"],
                documents=dataset["documents"],
                topk=args.topk,
            )
            metrics = compute_metrics(dataset["qrels"], retrieval_results)
            print("BM25 Retrieval Metrics:")
            for metric, value in metrics.items():
                print(f"{metric:<12}\t{value}")
        else:
            raise NotImplementedError(
                f"Retriever {args.retriever} not implemented yet, please use `bm25`.")

        rerank_documents = []
        rerank_doc_ids = []
        for query_id in dataset["queries"].keys():
            run = retrieval_results[query_id]
            rerank_doc_ids.append(
                sorted(run.keys(), key=lambda x: run[x], reverse=True)[:args.topk])
            rerank_documents.append(
                [dataset["documents"][doc_id] for doc_id in rerank_doc_ids[-1]])

        results = evaluate_one_dataset(
            reranker=reranker,
            queries=list(dataset["queries"].values()),
            query_ids=list(dataset["queries"].keys()),
            documents=rerank_documents,
            doc_ids=rerank_doc_ids,
            qrels=dataset["qrels"],
        )
        print("Reranking Metrics:")
        for metric, value in results.items():
            print(f"{metric:<12}\t{value}")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4, default=str)
    print(f"Results saved to {output_dir}")
