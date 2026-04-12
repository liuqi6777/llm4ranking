import copy
import math
import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Union, List
from itertools import combinations
from dataclasses import dataclass, field

from llm4ranking.lm.base import BatchLMOutput, LMOutput


@dataclass
class RankingRecord:

    query: str
    candidates: Optional[list[str]] = None

    prompt_template: Optional[str] = None
    num_processed_docs: Optional[int] = None
    latency: Optional[float] = None

    lm_outputs: list[Any] = field(default_factory=list)
    num_lm_calls: int = 0
    batch_sizes: list[int] = field(default_factory=list)

    rank_indices: Optional[list[int]] = None


def _chunk_sequence(items: list[Any], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


class Reranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable,
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any]
    ) -> tuple[list[str], list[int]]:
        pass


class PointwiseReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str], float],
        truncate_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:
        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=len(candidates),
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        if truncate_length:
            candidates = [" ".join(candidate.split()[:truncate_length]) for candidate in candidates]

        loglikelihoods = []
        effective_batch_size = len(candidates) if batch_size is None else max(batch_size, 1)
        use_batch = (
            len(candidates) > 0
            and getattr(ranking_func, "supports_batch", False)
            and hasattr(ranking_func, "score_many")
        )
        if use_batch:
            for candidate_batch in _chunk_sequence(candidates, effective_batch_size):
                if return_record:
                    batch_scores, lm_outputs = ranking_func.score_many(
                        query,
                        candidate_batch,
                        return_lm_outputs=True,
                        **kwargs,
                    )
                    record.lm_outputs.append(lm_outputs)
                    record.num_lm_calls += 1
                    record.batch_sizes.append(len(candidate_batch))
                else:
                    batch_scores = ranking_func.score_many(query, candidate_batch, **kwargs)
                loglikelihoods.extend(batch_scores)
        else:
            for candidate in candidates:
                if return_record:
                    loglikelihood, lm_outputs = ranking_func(query, candidate, return_lm_outputs=True, **kwargs)
                    record.lm_outputs.append(lm_outputs)
                    record.num_lm_calls += 1
                    record.batch_sizes.append(1)
                else:
                    loglikelihood = ranking_func(query, candidate, **kwargs)
                loglikelihoods.append(loglikelihood)

        if len(loglikelihoods) == 0:
            ranked_indices, ranked_result = [], []
        else:
            ranked_indices, _ = zip(*sorted(enumerate(loglikelihoods), key=lambda x: x[1], reverse=True))
            ranked_result = [candidates[i] for i in ranked_indices]

        outputs = (ranked_result,)
        if return_indices:
            outputs += (ranked_indices,)
        if return_record:
            record.latency = time.time() - t
            record.rank_indices = ranked_indices
            return outputs + (record,)
        return outputs


class PairwiseAllPairReranker(Reranker):

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        pair_batch_size: Optional[int] = None,
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:

        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=len(candidates),
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        doc_pairs = list(combinations(range(len(candidates)), 2))
        scores = [0] * len(candidates)
        effective_batch_size = len(doc_pairs) if pair_batch_size is None else max(pair_batch_size, 1)
        use_batch = (
            len(doc_pairs) > 0
            and getattr(ranking_func, "supports_batch", False)
            and hasattr(ranking_func, "compare_many")
        )
        if use_batch:
            for pair_batch in _chunk_sequence(doc_pairs, effective_batch_size):
                text_pairs = [(candidates[i], candidates[j]) for i, j in pair_batch]
                if return_record:
                    batch_scores, lm_outputs = ranking_func.compare_many(
                        query,
                        text_pairs,
                        return_lm_outputs=True,
                        **kwargs,
                    )
                    record.num_processed_docs += 4 * len(pair_batch)
                    record.lm_outputs.append(lm_outputs)
                    record.num_lm_calls += 2
                    record.batch_sizes.extend([len(pair_batch), len(pair_batch)])
                else:
                    batch_scores = ranking_func.compare_many(query, text_pairs, **kwargs)

                for (i, j), res in zip(pair_batch, batch_scores):
                    if res > 0:
                        scores[i] += 1
                    elif res < 0:
                        scores[j] += 1
                    else:
                        scores[i] += 0.5
                        scores[j] += 0.5
        else:
            for i, j in doc_pairs:
                if return_record:
                    res, lm_outputs = ranking_func(query, candidates[i], candidates[j], return_lm_outputs=True, **kwargs)
                    record.num_processed_docs += 4
                    record.lm_outputs.append(lm_outputs)
                    record.num_lm_calls += 2
                    record.batch_sizes.extend([1, 1])
                else:
                    res = ranking_func(query, candidates[i], candidates[j], **kwargs)

                if res > 0:
                    scores[i] += 1
                elif res < 0:
                    scores[j] += 1
                else:
                    scores[i] += 0.5
                    scores[j] += 0.5

        if len(scores) == 0:
            ranked_indices, ranked_result = [], []
        else:
            ranked_indices, _ = zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
            ranked_result = [candidates[i] for i in ranked_indices]

        outputs = (ranked_result,)
        if return_indices:
            outputs += (ranked_indices,)
        if return_record:
            record.latency = time.time() - t
            record.rank_indices = ranked_indices
            return outputs + (record,)
        return outputs


class PairwiseBubbleSortReranker(Reranker):

    def _compare_pair(
        self,
        query: str,
        left_doc: str,
        right_doc: str,
        ranking_func: Callable[[str, str, str], int],
        return_record: bool,
        record: Optional[RankingRecord],
        **kwargs,
    ):
        if return_record:
            res, lm_outputs = ranking_func(query, left_doc, right_doc, return_lm_outputs=True, **kwargs)
            record.num_processed_docs += 4
            record.lm_outputs.append(lm_outputs)
            record.num_lm_calls += 2
            record.batch_sizes.extend([1, 1])
            return res
        return ranking_func(query, left_doc, right_doc, **kwargs)

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        topk: int = 10,
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:

        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=len(candidates),
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        last_end = len(candidates) - 1
        ranked_indices = list(range(len(candidates)))
        for i in range(min(topk, len(candidates))):
            changed = False
            for j in range(last_end, i, -1):
                res = self._compare_pair(
                    query=query,
                    left_doc=candidates[j],
                    right_doc=candidates[j - 1],
                    ranking_func=ranking_func,
                    return_record=return_record,
                    record=record if return_record else None,
                    **kwargs,
                )

                if res > 0:
                    candidates[j - 1], candidates[j] = candidates[j], candidates[j - 1]
                    ranked_indices[j - 1], ranked_indices[j] = ranked_indices[j], ranked_indices[j - 1]
                    if not changed:
                        changed = True
                        # skip unchanged pairs at the bottom
                        if last_end != len(candidates) - 1:
                            last_end += 1
                if not changed:
                    last_end -= 1

        outputs = (candidates,)
        if return_indices:
            outputs += (ranked_indices,)
        if return_record:
            record.latency = time.time() - t
            record.rank_indices = ranked_indices
            return outputs + (record,)
        return outputs


class PairwiseHeapSortReranker(Reranker):

    def _compare_pair(
        self,
        query: str,
        left_doc: str,
        right_doc: str,
        ranking_func: Callable[[str, str, str], int],
        return_record: bool,
        record: Optional[RankingRecord],
        **kwargs,
    ):
        if return_record:
            res, lm_outputs = ranking_func(query, left_doc, right_doc, return_lm_outputs=True, **kwargs)
            record.num_processed_docs += 4
            record.lm_outputs.append(lm_outputs)
            record.num_lm_calls += 2
            record.batch_sizes.extend([1, 1])
            return res
        return ranking_func(query, left_doc, right_doc, **kwargs)

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        topk: int = 10,
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:

        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=0,
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        ranked_indices = list(range(len(candidates)))

        def wrapped_ranking_func(i, j):
            return self._compare_pair(
                query=query,
                left_doc=candidates[i],
                right_doc=candidates[j],
                ranking_func=ranking_func,
                return_record=return_record,
                record=record if return_record else None,
                **kwargs,
            )

        def heapify(n, i):
            # Find largest among root and children
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            if l < n and wrapped_ranking_func(l, i) > 0:
                largest = l
            if r < n and wrapped_ranking_func(r, largest) > 0:
                largest = r
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                candidates[i], candidates[largest] = candidates[largest], candidates[i]
                ranked_indices[i], ranked_indices[largest] = ranked_indices[largest], ranked_indices[i]
                heapify(n, largest)

        n = len(candidates)
        # Build max heap
        for i in range(n // 2, -1, -1):
            heapify(n, i)
        for i in range(n - 1, max(n - 1 - topk, 0), -1):
            candidates[i], candidates[0] = candidates[0], candidates[i]
            ranked_indices[i], ranked_indices[0] = ranked_indices[0], ranked_indices[i]
            # Heapify root element
            heapify(i, 0)

        outputs = (reversed(candidates),)
        if return_indices:
            outputs += (reversed(ranked_indices),)
        if return_record:
            record.latency = time.time() - t
            record.rank_indices = reversed(ranked_indices)
            return outputs + (record,)
        return outputs


class ListwiseSilidingWindowReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, list[str]], list[int]],
        rank_start: int = 0,
        rank_end: int = None,
        window_size: Optional[int] = None,
        step: int = 10,
        truncate_length: Optional[int] = None,
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:

        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=0,
                prompt_template=ranking_func._prompt_template,
            )
        t = time.time()

        if truncate_length:
            candidates = [" ".join(candidate.split()[:truncate_length]) for candidate in candidates]

        ranked_result = copy.deepcopy(candidates)
        ranked_indices = list(range(len(candidates)))

        window_size = window_size or len(candidates)
        rank_end = rank_end or len(candidates)
        start_pos, end_pos = rank_end - window_size, rank_end
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            # range from 0 to window_size

            if return_record:
                permutation, lm_outputs = ranking_func(query, ranked_result[start_pos:end_pos], return_lm_outputs=True, **kwargs)
                record.num_processed_docs += (end_pos - start_pos)
                record.lm_outputs.append(lm_outputs)
            else:
                permutation = ranking_func(query, ranked_result[start_pos:end_pos], **kwargs)

            # receive permutation
            cut_range = copy.deepcopy(ranked_result[start_pos:end_pos])
            cut_range_indices = copy.deepcopy(ranked_indices[start_pos:end_pos])
            for local_rank, index in enumerate(permutation):
                ranked_result[start_pos + local_rank] = copy.deepcopy(cut_range[index])
                ranked_indices[start_pos + local_rank] = cut_range_indices[index]

            start_pos, end_pos = start_pos - step, end_pos - step

        outputs = (ranked_result,)
        if return_indices:
            outputs += (ranked_indices,)
        if return_record:
            record.latency = time.time() - t
            record.rank_indices = ranked_indices
            return outputs + (record,)
        return outputs


class TournamentReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str], float],
        tuornament_times: int = 1,
        group_sizes: tuple[int] = (10, 10, 10, 10, 5),
        promotion_sizes: tuple[int] = (5, 4, 5, 5, 2),
        stage_weight: float = 0.,
        truncate_length: Optional[int] = None,
        return_record: bool = False, 
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:

        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=0,
                prompt_template=ranking_func._prompt_template,
            )
        t = time.time()

        assert len(group_sizes) == len(promotion_sizes)

        if truncate_length:
            candidates = [" ".join(candidate.split()[:truncate_length]) for candidate in candidates]

        if len(candidates) == 0:
            outputs = ([], )
            if return_indices:
                outputs += ([], )
            if return_record:
                outputs += (None, )
            return outputs

        doc_scores = [0] * len(candidates)

        for _ in range(tuornament_times):
            # construct a random order of documents for the first stage
            stage_ids = list(range(len(candidates)))

            for stage, (group_size, promotion_size) in enumerate(zip(group_sizes, promotion_sizes)):

                num_groups = max(1, math.ceil(len(stage_ids) / group_size))
                groups = [[] for _ in range(num_groups)]
                for idx, v in enumerate(stage_ids):
                    groups[idx % num_groups].append(v)

                new_stage = []
                for group_candidate_ids in groups:
                    if not group_candidate_ids:
                        continue

                    random.shuffle(group_candidate_ids)
                    group_candidates = [candidates[i] for i in group_candidate_ids]
                    k = min(len(group_candidate_ids), promotion_size)

                    if k == 0:
                        continue

                    if return_record:
                        top_indices, lm_outputs = ranking_func(query, group_candidates, min(promotion_size, len(group_candidates)), return_lm_outputs=True, **kwargs)
                        record.num_processed_docs += len(group_candidates)
                        record.lm_outputs.append(lm_outputs)
                    else:
                        top_indices = ranking_func(query, group_candidates, min(promotion_size, len(group_candidates)), **kwargs)  # select top M form N candidates

                    global_ids = [group_candidate_ids[i] for i in top_indices]
                    new_stage.extend(global_ids)

                    for gid in global_ids:
                        doc_scores[gid] += (1 + stage * stage_weight)

                stage_ids = sorted(new_stage, key=lambda x: doc_scores[x], reverse=True)

                if len(stage_ids) <= 1:
                    break

        ranked_indices, _ = zip(*sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True))
        ranked_result = [candidates[i] for i in ranked_indices]

        outputs = (ranked_result,)
        if return_indices:
            outputs += (ranked_indices,)
        if return_record:
            record.latency = time.time() - t
            record.rank_indices = ranked_indices
            return outputs + (record,)
        return outputs
