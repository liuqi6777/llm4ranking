import argparse
import datetime
import json
import os

from llm4ranking import Reranker
from llm4ranking.evaluation.evaluator import evaluate_one_dataset
from llm4ranking.evaluation.utils import load_mair, retrieval_bm25
from llm4ranking.evaluation.trec_eval import trec_eval, compute_metrics


TASK_CONFIG = {'Competition-Math': 'Academic', 'ProofWiki_Proof': 'Academic', 'ProofWiki_Reference': 'Academic',
               'Stacks_Proof': 'Academic', 'Stacks_Reference': 'Academic', 'Stein_Proof': 'Academic',
               'Stein_Reference': 'Academic', 'Trench_Proof': 'Academic', 'Trench_Reference': 'Academic',
               'TAD': 'Academic', 'TAS2': 'Academic', 'StackMathQA': 'Academic', 'APPS': 'Code',
               'CodeEditSearch': 'Code', 'CodeSearchNet': 'Code', 'Conala': 'Code', 'HumanEval-X': 'Code',
               'LeetCode': 'Code', 'MBPP': 'Code', 'RepoBench': 'Code', 'TLDR': 'Code', 'SWE-Bench-Lite': 'Code',
               'Apple': 'Finance', 'ConvFinQA': 'Finance', 'FinQA': 'Finance', 'FinanceBench': 'Finance',
               'HC3Finance': 'Finance', 'TAT-DQA': 'Finance', 'Trade-the-event': 'Finance', 'AY2': 'Web', 'ELI5': 'Web',
               'Fever': 'Web', 'TREx': 'Web', 'WnCw': 'Web', 'WnWi': 'Web', 'WoW': 'Web', 'zsRE': 'Web',
               'AILA2019-Case': 'Legal', 'AILA2019-Statutes': 'Legal', 'BSARD': 'Legal', 'BillSum': 'Legal',
               'CUAD': 'Legal', 'GerDaLIR': 'Legal', 'LeCaRDv2': 'Legal', 'LegalQuAD': 'Legal', 'REGIR-EU2UK': 'Legal',
               'REGIR-UK2EU': 'Legal', 'ArguAna': 'Web', 'CQADupStack': 'Web', 'FiQA': 'Finance', 'NFCorpus': 'Medical',
               'Quora': 'Web', 'SciDocs': 'Academic', 'SciFact': 'Academic', 'TopiOCQA': 'Web', 'Touche': 'Web',
               'Trec-Covid': 'Medical', 'ACORDAR': 'Web', 'CPCD': 'Web', 'ChroniclingAmericaQA': 'Web',
               'Monant': 'Medical', 'NTCIR': 'Web', 'PointRec': 'Web', 'ProCIS-Dialog': 'Web', 'ProCIS-Turn': 'Web',
               'QuanTemp': 'Web', 'WebTableSearch': 'Web', 'CARE': 'Medical', 'MISeD': 'Web', 'SParC': 'Web',
               'SParC-SQL': 'Web', 'Spider': 'Web', 'Spider-SQL': 'Web', 'LitSearch': 'Academic', 'CAsT_2019': 'Web',
               'CAsT_2020': 'Web', 'CAsT_2021': 'Web', 'CAsT_2022': 'Web', 'Core_2017': 'Web', 'Microblog_2011': 'Web',
               'Microblog_2012': 'Web', 'Microblog_2013': 'Web', 'Microblog_2014': 'Web',
               'PrecisionMedicine_2017': 'Medical', 'PrecisionMedicine_2018': 'Medical',
               'PrecisionMedicine_2019': 'Medical', 'PrecisionMedicine-Article_2019': 'Medical',
               'PrecisionMedicine-Article_2020': 'Medical', 'CliniDS_2014': 'Medical', 'CliniDS_2015': 'Medical',
               'CliniDS_2016': 'Medical', 'ClinicalTrials_2021': 'Medical', 'ClinicalTrials_2022': 'Medical',
               'ClinicalTrials_2023': 'Medical', 'DD_2015': 'Web', 'DD_2016': 'Web', 'DD_2017': 'Web',
               'FairRanking_2020': 'Academic', 'FairRanking_2021': 'Web', 'FairRanking_2022': 'Web',
               'Genomics-AdHoc_2004': 'Medical', 'Genomics-AdHoc_2005': 'Medical', 'Genomics-AdHoc_2006': 'Medical',
               'Genomics-AdHoc_2007': 'Medical', 'TREC-Legal_2011': 'Legal', 'NeuCLIR-Tech_2023': 'Web',
               'NeuCLIR_2022': 'Web', 'NeuCLIR_2023': 'Web', 'ProductSearch_2023': 'Web', 'ToT_2023': 'Web',
               'ToT_2024': 'Web', 'FoodAPI': 'Code', 'HuggingfaceAPI': 'Code', 'PytorchAPI': 'Code',
               'SpotifyAPI': 'Code', 'TMDB': 'Code', 'TensorAPI': 'Code', 'ToolBench': 'Code', 'WeatherAPI': 'Code',
               'ExcluIR': 'Web', 'Core17': 'Web', 'News21': 'Web', 'Robust04': 'Web', 'InstructIR': 'Web',
               'NevIR': 'Web', 'IFEval': 'Web'}


# get all tasks of a domain (Academic, Code, Web, Legal, Medical, Finance)
def get_tasks_by_domain(domain):
    assert domain in ['Academic', 'Code', 'Web', 'Legal', 'Medical', 'Finance']
    out = []
    for task in TASK_CONFIG:
        if TASK_CONFIG[task] == domain:
            out.append(task)
    return out


# return list of all tasks
def get_all_tasks():
    return list(TASK_CONFIG.keys())



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
        dataset = load_mair(task, instruct=True)
        for split in dataset.keys():
            print(f"Evaluating split: {split}")
            if args.retriever == "bm25":
                retrieval_results = retrieval_bm25(
                    queries=dataset[split]["queries"],
                    documents=dataset[split]["documents"],
                    topk=args.topk,
                )
                metrics = compute_metrics(dataset[split]["qrels"], retrieval_results)
                print("BM25 Retrieval Metrics:")
                for metric, value in metrics.items():
                    print(f"{metric:<12}\t{value}")
            else:
                raise NotImplementedError(
                    f"Retriever {args.retriever} not implemented yet, please use `bm25`.")

            rerank_documents = []
            rerank_doc_ids = []
            for query_id in dataset[split]["queries"].keys():
                run = retrieval_results[query_id]
                rerank_doc_ids.append(
                    sorted(run.keys(), key=lambda x: run[x], reverse=True)[:args.topk])
                rerank_documents.append(
                    [dataset[split]["documents"][doc_id] for doc_id in rerank_doc_ids[-1]])

            results = evaluate_one_dataset(
                reranker=reranker,
                queries=list(dataset[split]["queries"].values()),
                query_ids=list(dataset[split]["queries"].keys()),
                documents=rerank_documents,
                doc_ids=rerank_doc_ids,
                qrels=dataset[split]["qrels"],
            )

            all_results[split] = results
            print(f"Results for {split}:")
            for metric, value in results.items():
                print(f"{metric:<12}\t{value}") 

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    print(f"Results saved to {output_dir}")
