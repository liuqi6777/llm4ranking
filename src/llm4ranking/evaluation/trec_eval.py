import argparse
import collections
import pytrec_eval

from datasets import load_dataset


INDEX = {
    'bm25': {
        'dl19': 'msmarco-v1-passage',
        'dl20': 'msmarco-v1-passage',
        'covid': 'beir-v1.0.0-trec-covid.flat',
        'arguana': 'beir-v1.0.0-arguana.flat',
        'touche': 'beir-v1.0.0-webis-touche2020.flat',
        'news': 'beir-v1.0.0-trec-news.flat',
        'scifact': 'beir-v1.0.0-scifact.flat',
        'fiqa': 'beir-v1.0.0-fiqa.flat',
        'scidocs': 'beir-v1.0.0-scidocs.flat',
        'nfc': 'beir-v1.0.0-nfcorpus.flat',
        'quora': 'beir-v1.0.0-quora.flat',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
        'fever': 'beir-v1.0.0-fever.flat',
        'robust04': 'beir-v1.0.0-robust04.flat',
        'signal': 'beir-v1.0.0-signal1m.flat',
        'nq': 'beir-v1.0.0-nq.flat',
        'cfever': 'beir-v1.0.0-climate-fever.flat',
        'hotpotqa': 'beir-v1.0.0-hotpotqa.flat',
        'mrtydi-ar': 'mrtydi-v1.1-arabic',
        'mrtydi-bn': 'mrtydi-v1.1-bengali',
        'mrtydi-fi': 'mrtydi-v1.1-finnish',
        'mrtydi-id': 'mrtydi-v1.1-indonesian',
        'mrtydi-ja': 'mrtydi-v1.1-japanese',
        'mrtydi-ko': 'mrtydi-v1.1-korean',
        'mrtydi-ru': 'mrtydi-v1.1-russian',
        'mrtydi-sw': 'mrtydi-v1.1-swahili',
        'mrtydi-te': 'mrtydi-v1.1-telugu',
        'mrtydi-th': 'mrtydi-v1.1-thai',
    },
    'splade++ed': {
        'dl19': 'msmarco-v1-passage-splade-pp-ed-text',
        'dl20': 'msmarco-v1-passage-splade-pp-ed-text',
        'covid': 'beir-v1.0.0-trec-covid.splade-pp-ed',
        'arguana': 'beir-v1.0.0-arguana.splade-pp-ed',
        'touche': 'beir-v1.0.0-webis-touche2020.splade-pp-ed',
        'news': 'beir-v1.0.0-trec-news.splade-pp-ed',
        'scifact': 'beir-v1.0.0-scifact.splade-pp-ed',
        'fiqa': 'beir-v1.0.0-fiqa.splade-pp-ed',
        'scidocs': 'beir-v1.0.0-scidocs.splade-pp-ed',
        'nfc': 'beir-v1.0.0-nfcorpus.splade-pp-ed',
        'quora': 'beir-v1.0.0-quora.splade-pp-ed',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity.splade-pp-ed',
        'fever': 'beir-v1.0.0-fever.splade-pp-ed',
        'robust04': 'beir-v1.0.0-robust04.splade-pp-ed',
        'signal': 'beir-v1.0.0-signal1m.splade-pp-ed',
        'nq': 'beir-v1.0.0-nq.splade-pp-ed',
        'cfever': 'beir-v1.0.0-climate-fever.splade-pp-ed',
        'hotpotqa': 'beir-v1.0.0-hotpotqa.splade-pp-ed',
    },
}

TOPICS_AND_QRELS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    # BEIR
    'covid': 'beir-v1.0.0-trec-covid.test',
    'arguana': 'beir-v1.0.0-arguana.test',
    'touche': 'beir-v1.0.0-webis-touche2020.test',
    'news': 'beir-v1.0.0-trec-news.test',
    'scifact': 'beir-v1.0.0-scifact.test',
    'fiqa': 'beir-v1.0.0-fiqa.test',
    'scidocs': 'beir-v1.0.0-scidocs.test',
    'nfc': 'beir-v1.0.0-nfcorpus.test',
    'quora': 'beir-v1.0.0-quora.test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.test',
    'fever': 'beir-v1.0.0-fever.test',
    'robust04': 'beir-v1.0.0-robust04.test',
    'signal': 'beir-v1.0.0-signal1m.test',
    'nq': 'beir-v1.0.0-nq.test',
    'cfever': 'beir-v1.0.0-climate-fever.test',
    'hotpotqa': 'beir-v1.0.0-hotpotqa.test',
    # Mr.Tydi
    'mrtydi-ar': 'mrtydi-v1.1-ar.test',
    'mrtydi-bn': 'mrtydi-v1.1-bn.test',
    'mrtydi-fi': 'mrtydi-v1.1-fi.test',
    'mrtydi-id': 'mrtydi-v1.1-id.test',
    'mrtydi-ja': 'mrtydi-v1.1-ja.test',
    'mrtydi-ko': 'mrtydi-v1.1-ko.test',
    'mrtydi-ru': 'mrtydi-v1.1-ru.test',
    'mrtydi-sw': 'mrtydi-v1.1-sw.test',
    'mrtydi-te': 'mrtydi-v1.1-te.test',
    'mrtydi-th': 'mrtydi-v1.1-th.test',
    # BRIGHT
    'bright-biology': 'bright-biology.test',
    'bright-earth-science': 'bright-earth-science.test',
    'bright-economics': 'bright-economics.test',
    'bright-psychology': 'bright-psychology.test',
    'bright-robotics': 'bright-robotics.test',
    'bright-stackoverflow': 'bright-stackoverflow.test',
    'bright-sustainable-living': 'bright-sustainable-living.test',
    'bright-pony': 'bright-pony.test',
    'bright-leetcode': 'bright-leetcode.test',
    'bright-aops': 'bright-aops.test',
    'bright-theoremqa-theorems': 'bright-theoremqa-theorems.test',
    'bright-theoremqa-questions': 'bright-theoremqa-questions.test',
    # BrowseComp-Plus
    "browsecomp-plus": "browsecomp-plus.test",
}


def get_qrels(dataset: str) -> str:
    qrel = collections.defaultdict(dict)
    f_qrel = load_dataset(
        "liuqi6777/retrieval_results",
        data_files=f"topics_and_qrels/qrels.{TOPICS_AND_QRELS[dataset]}.txt",
        split="train"
    )["text"]

    for line in f_qrel:
        query_id, _, object_id, relevance = line.strip().split()

        assert object_id not in qrel[query_id]
        qrel[query_id][object_id] = int(relevance)

    return qrel


def compute_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: tuple[int] = (10, 50, 100, 200, 1000)
) -> dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        _map[f"MAP@{k}"] = 0.0
        ndcg[f"NDCG@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 4) for k, v in m.items()}

    _map = _normalize(_map)
    ndcg = _normalize(ndcg)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [_map, ndcg, recall]:
        all_metrics.update(mt)

    return all_metrics


def filter_run(run, excluded_ids):
    new_run = collections.defaultdict(dict)
    for qid in run.keys():
        for docid, score in run[qid].items():
            if docid in excluded_ids.get(qid, ["N/A"]) and docid != "N/A":
                continue
            else:
                new_run[qid][docid] = score
    return new_run


def trec_eval(dataset, ranking, excluded_ids=None, print_metrics=True):
    with open(ranking, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    if excluded_ids is not None:
        run = filter_run(run, excluded_ids)
    qrels = get_qrels(dataset)
    all_metrics = compute_metrics(qrels, run, k_values=(1, 5, 10, 20, 100))
    if print_metrics:
        for metric, value in all_metrics.items():
            print(f"{metric:<12}\t{value}")
    return all_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dl19')
    parser.add_argument('--ranking', type=str, required=True)
    args = parser.parse_args()
    trec_eval(args.dataset, args.ranking)
