import argparse
import json
import os
from tqdm import tqdm

from llm4ranking.ranker.base import ListwiseSilidingWindowReranker, PointwiseReranker
from llm4ranking.model.listwise_llm import ListwiseRanker
from llm4ranking.model.relevance_generation import RelevanceGenerationRanker


def write_results(rerank_results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for i, hits in enumerate(rerank_results):
            for j, hit in enumerate(hits):
                f.write(
                    f"{hit['qid']} Q{i} {hit['docid']} {j + 1} {round(1 / (j + 1), 3)} rank")
                f.write("\n")


def eval_model(args):
    from trec_eval import trec_eval, TOPICS

    reranker = ListwiseSilidingWindowReranker()
    ranking_model = ListwiseRanker(args.model_path)

    for dataset in args.datasets:

        output_file = os.path.join(
            "results", "rerank_results", args.retriever,
            f"eval_{dataset}_{ranking_model.model_name.split('/')[-1]}_top{args.topk}.txt"
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

        if args.num_passes == 1:
            rerank_results = []
            for i in tqdm(range(len(data))):
                rerank_result = reranker.rerank(
                    query=data[i]["query"],
                    candidates=data[i]["hits"],
                    ranking_func=ranking_model,
                    window_size=20,
                    step=10,
                )
                rerank_results.append(rerank_result)
            write_results(rerank_results, output_file)
            trec_eval(TOPICS[dataset], output_file)
        else:
            retrieval_results = [data[i]["hits"] for i in range(len(data))]
            for pass_ in range(args.num_passes):
                rerank_results = []
                for i in tqdm(range(len(data))):
                    rerank_result = reranker.rerank(
                        query=data[i]["query"],
                        candidates=retrieval_results[i],
                        ranking_func=ranking_model,
                        window_size=20,
                        step=10,
                    )
                    rerank_results.append(rerank_result)
                retrieval_results = rerank_results
                write_results(rerank_results, output_file.replace(".txt", f"_pass{pass_}.txt"))
                trec_eval(TOPICS[dataset], output_file.replace(".txt", f"_pass{pass_}.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--datasets", nargs="+", default=["dl19"])
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--num-passes", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    eval_model(args)
