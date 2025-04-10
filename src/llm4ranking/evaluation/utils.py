import bm25s

from tqdm import tqdm
from typing import Dict
from datasets import load_dataset


def retrieval_bm25(
    queries: Dict[str, str],
    documents: Dict[str, str],
    topk: int = 100,
):
    doc_ids, documents = zip(*documents.items())
    doc_ids, documents = list(doc_ids), list(documents)
    print(f"Number of documents: {len(documents)}")
    print(f"Number of queries: {len(queries)}")
    print("Tokenizing documents...")
    corpus_tokens = bm25s.tokenize(documents, stopwords="en")
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    print("Indexing documents...")
    retriever.index(corpus_tokens)

    results = {}

    print("Retrieving documents...")
    for query_id, query in tqdm(queries.items()):

        query_tokens = bm25s.tokenize(query)
        if len(query_tokens.vocab) == 0:
            query_tokens = bm25s.tokenize("NONE", stopwords=[])

        hits, scores = retriever.retrieve(query_tokens, corpus=doc_ids, k=min(topk, len(doc_ids)))
        
        results[query_id] = {}
        for i in range(len(hits[0])):
            results[query_id][hits[0, i]] = float(scores[0, i])

    return results


def load_mair(task, instruct=True):
    all_queries = load_dataset("MAIR-Bench/MAIR-Queries", task)
    all_docs = load_dataset("MAIR-Bench/MAIR-Docs", task)

    all_results = {}
    for split in all_queries:
        queries_split = all_queries[split]
        if split == "queries":
            docs_split = all_docs["docs"]
        else:
            docs_split = all_docs[split.replace("_queries", "_docs")]

        queries = {}
        for item in queries_split:
            query = item["query"]
            if instruct:
                query = item["instruction"] + " " + query
            queries[item["qid"]] = query

        qrels = {}
        for item in queries_split:
            qrels[item["qid"]] = {str(x["id"]): int(x["score"]) for x in item["labels"]}

        all_results[split] = {
            "queries": queries,
            "documents": {item["id"]: item["doc"] for item in docs_split},
            "qrels": qrels,
        }
    return all_results


def load_bright(task, long_context=False, reasoning=None):
    if reasoning:
        all_queries = load_dataset("xlangai/bright", f"{reasoning}_reason")[task]
    else:
        all_queries = load_dataset("xlangai/bright", "examples")[task]
    if long_context:
        all_docs = load_dataset("xlangai/bright", "long_documents")[task]
    else:
        all_docs = load_dataset("xlangai/bright", "documents")[task]
    documents = {doc["id"]: doc["content"] for doc in all_docs}

    queries = {}
    excluded_ids = {}

    for item in all_queries:
        queries[item["id"]] = item["query"]
        if "excluded_ids" in item:
            excluded_ids[item["id"]] = item["excluded_ids"]

    qrels = {}
    if long_context:
        for item in all_queries:
            qrels[item["id"]] = {docid: 1 for docid in item["gold_ids_long"]}
    else:
        for item in all_queries:
            qrels[item["id"]] = {docid: 1 for docid in item["gold_ids"]}

    return {
        "queries": queries,
        "documents": documents,
        "qrels": qrels,
        "excluded_ids": excluded_ids,
    }
