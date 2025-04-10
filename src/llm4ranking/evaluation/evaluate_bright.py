import argparse
import datetime
import json
import os

from llm4ranking import Reranker
from llm4ranking.evaluation.evaluator import evaluate_one_dataset
from llm4ranking.evaluation.utils import load_bright, retrieval_bm25
from llm4ranking.evaluation.trec_eval import trec_eval, compute_metrics


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
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--reranking_args", type=parse_dict_args, default={})
    parser.add_argument("--model_fw_args", type=parse_dict_args, default={})
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    print(args)

    if args.output_dir is None:
        output_dir = os.path.join(
            "results", "runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cli_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    reranker = Reranker(
        reranking_approach=args.reranking_approach,
        model_type=args.model_type,
        model_name=args.model_args["model"],
        model_args=args.model_args,
        reranking_args=args.reranking_args,
        model_fw_args=args.model_fw_args,
        prompt_template=args.prompt_template,
    )

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
