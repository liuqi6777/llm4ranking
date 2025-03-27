import copy
import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Union, List
from itertools import combinations
from dataclasses import dataclass, field

from llm4ranking.lm.base import LMOuput


@dataclass
class RankingRecord:

    query: str
    candidates: Optional[list[str]] = None

    prompt_template: Optional[str] = None
    num_processed_docs: Optional[int] = None
    num_processed_tokens: Optional[int] = None
    num_generated_tokens: Optional[int] = None
    latency: Optional[float] = None

    lm_outputs: list[LMOuput] = field(default_factory=list)

    rank_indices: Optional[list[int]] = None


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
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:
        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=len(candidates),
                num_processed_tokens=0,
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        if truncate_length:
            candidates = [" ".join(candidate.split()[:truncate_length]) for candidate in candidates]

        loglikelihoods = []
        for candidate in candidates:
            if return_record:
                loglikelihood, lm_outputs = ranking_func(query, candidate, return_lm_outputs=True, **kwargs)
                record.num_processed_tokens += lm_outputs.num_processed_tokens
                record.lm_outputs.append(lm_outputs)
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
        return_record: bool = False,
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:

        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=len(candidates),
                num_processed_tokens=0,
                num_generated_tokens=0,
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        doc_pairs = list(combinations(range(len(candidates)), 2))
        scores = [0] * len(candidates)
        for i, j in doc_pairs:

            if return_record:
                res, lm_outputs = ranking_func(query, candidates[i], candidates[j], return_lm_outputs=True, **kwargs)
                record.num_processed_tokens += lm_outputs.num_processed_tokens
                record.num_generated_tokens += lm_outputs.num_generated_tokens
                record.num_processed_docs += 4
            else:
                res = ranking_func(query, candidates[i], candidates[j], **kwargs)

            if res > 0:
                scores[i] += 1
            elif res < 0:
                scores[j] += 1
            else:
                scores[i] += 0.5
                scores[j] += 0.5

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
                num_processed_tokens=0,
                num_generated_tokens=0,
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        last_end = len(candidates) - 1
        ranked_indices = list(range(len(candidates)))
        for i in range(min(topk, len(candidates))):
            changed = False
            for j in range(last_end, i, -1):

                if return_record:
                    res, lm_outputs = ranking_func(query, candidates[j], candidates[j - 1], return_lm_outputs=True, **kwargs)
                    record.num_processed_tokens += lm_outputs.num_processed_tokens
                    record.num_generated_tokens += lm_outputs.num_generated_tokens
                    record.num_processed_docs += 4
                else:
                    res = ranking_func(query, candidates[j], candidates[j - 1], **kwargs)

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
                num_processed_tokens=0,
                num_generated_tokens=0,
                prompt_template=ranking_func._prompt_template,
            )
            t = time.time()

        ranked_indices = list(range(len(candidates)))

        def wrapped_ranking_func(i, j):
            if return_record:
                res, lm_outputs = ranking_func(query, candidates[i], candidates[j], return_lm_outputs=True, **kwargs)
                record.num_processed_tokens += lm_outputs.num_processed_tokens
                record.num_generated_tokens += lm_outputs.num_generated_tokens
                record.num_processed_docs += 4
            else:
                res = ranking_func(query, candidates[i], candidates[j], **kwargs)
            return res

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
                num_processed_tokens=0,
                num_generated_tokens=0,
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
                record.num_processed_tokens += lm_outputs.num_processed_tokens
                record.num_generated_tokens += lm_outputs.num_generated_tokens
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
        group_sizes: tuple[int] = (20, 10, 20, 10, 5),
        promotion_sizes: tuple[int] = (10, 4, 10, 5, 1),
        truncate_length: Optional[int] = None,
        return_record: bool = False, 
        return_indices: bool = False,
        **kwargs: dict[str, Any],
    ) -> Union[tuple[list[str], list[int]], tuple[list[str], list[int], RankingRecord]]:

        if return_record:
            record = RankingRecord(
                query=query,
                num_processed_docs=0,
                num_processed_tokens=0,
                num_generated_tokens=0,
                prompt_template=ranking_func._prompt_template,
            )
        t = time.time()

        assert len(group_sizes) == len(promotion_sizes)

        if truncate_length:
            candidates = [" ".join(candidate.split()[:truncate_length]) for candidate in candidates]

        doc_scores = [0] * len(candidates)

        # construct a random order of documents for the first stage
        stage_ids = list(range(len(candidates)))

        for _ in range(tuornament_times):
            for group_size, promotion_size in zip(group_sizes, promotion_sizes):
                assert group_size > promotion_size and len(stage_ids) % group_size == 0

                num_groups = len(stage_ids) // group_size
                groups = [stage_ids[i::num_groups] for i in range(num_groups)]

                stage_ids = []
                for gid in range(len(groups)):

                    group_candidate_ids = groups[gid]
                    random.shuffle(group_candidate_ids)
                    group_candidates = [candidates[i] for i in group_candidate_ids]

                    if return_record:
                        top_indices, lm_outputs = ranking_func(query, group_candidates, promotion_size, return_lm_outputs=True, **kwargs)
                        record.num_processed_docs += len(group_candidates)
                        record.num_processed_tokens += lm_outputs.num_processed_tokens
                        record.num_generated_tokens += lm_outputs.num_generated_tokens
                        record.lm_outputs.append(lm_outputs)
                    else:
                        top_indices = ranking_func(query, group_candidates, promotion_size, **kwargs)  # select top M form N candidates

                    top_ids = [group_candidate_ids[i] for i in top_indices]  # convert to global index
                    stage_ids.extend(top_ids)

                    for doc_id in top_ids:
                        doc_scores[doc_id] += 1

                stage_ids = sorted(stage_ids, key=lambda x: doc_scores[x], reverse=True)

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
