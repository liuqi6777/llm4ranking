import argparse
import json
import os
from functools import partial
from tqdm import tqdm

from llm4ranking.ranker import *
from llm4ranking.model import *
from llm4ranking.evaluation.trec_eval import trec_eval, TOPICS

def get_reranking_function(args):
    if args.reranking_approach == "listwise":
        reranker = ListwiseSilidingWindowReranker()
        ranking_func = ListwiseGeneration(args.model_path)
        return partial(reranker.rerank, ranking_func=ranking_func, window_size=args.window_size, step=args.step)
    elif args.reranking_approach == "pointwise":
        reranker = PointwiseReranker()
        if args.method == "relevance_generation":
            ranking_func = RelevanceGeneration(args.model_path)
        elif args.method == "query_generation":
            ranking_func = QueryGeneration(args.model_path)
        else:
            raise ValueError
        return partial(reranker.rerank, ranking_func=ranking_func)
    elif args.reranking_approach == "pairwise":
        ranking_func = PairwiseComparison(args.model_path)
        reranker = PairwiseReranker()
        return partial(reranker.rerank, ranking_func=ranking_func, method=args.method)
    else:
        raise ValueError


def write_results(rerank_results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for i, item in enumerate(rerank_results):
            hits = item["hits"]
            for j, hit in enumerate(hits):
                f.write(
                    f"{hit['qid']} Q{i} {hit['docid']} {j + 1} {round(1 / (j + 1), 3)} rank")
                f.write("\n")


def eval_model(args):

    rerank = get_reranking_function(args)

    for dataset in args.datasets:

        output_file = os.path.join(
            "results", "rerank_results", args.retriever,
            f"eval_{dataset}_{args.model_path.split('/')[-1]}_top{args.topk}.txt"
        )
        if os.path.exists(output_file) and not args.overwrite:
            print(f"{output_file} exists, skipping")
            continue

        input_file = os.path.join(
            "results", "retrieval_results", args.retriever,
            f"{dataset}_top{args.topk}.jsonl"
        )
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]

        prev_results = data
        for pass_ in range(args.num_passes):
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
            write_results(rerank_results, output_file.replace(".txt", f"_pass{pass_}.txt"))
            trec_eval(TOPICS[dataset], output_file.replace(".txt", f"_pass{pass_}.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", nargs="+", default=["dl19"])
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--reranking-approach", type=str, choices=["listwise", "pairwise", "pointwise", "tourrank"], default="listwise")

    # subparsers for different reranking approaches
    subparsers = parser.add_subparsers()
    # for listwise approach
    listwise_parser = subparsers.add_parser("listwise-params")
    listwise_parser.add_argument("--window-size", type=int, default=20)
    listwise_parser.add_argument("--step", type=int, default=10)
    # for pointwise approach
    pointwise_parser = subparsers.add_parser("pointwise-params")
    pointwise_parser.set_defaults(method="relevance_generation")
    pointwise_parser.add_argument("--method", type=str, choices=["relevance_generation", "query_generation"], default="relevance_generation")
    # for pairwise approach
    pairwise_parser = subparsers.add_parser("pairwise-params")
    pairwise_parser.add_argument("--method", type=str, choices=["allpair", "bubble_sort", "heap_sort"], default="heap_sort")
    # for tour rank approach
    tourrank_parser = subparsers.add_parser("tourrank-params")

    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--num-passes", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    eval_model(args)
