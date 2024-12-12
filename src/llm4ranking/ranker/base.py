import copy
import random
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from itertools import combinations


class Reranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable,
        **kwargs: dict[str, Any]
    ) -> tuple[list[str], list[int]]:
        pass


class PointwiseReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str], float],
        **kwargs: dict[str, Any],
    ) -> tuple[list[str], list[int]]:
        ranked_indices, ranked_result = zip(*sorted(enumerate(candidates), key=lambda x: ranking_func(query, x[1], **kwargs), reverse=True))
        return ranked_result, ranked_indices


class PairwiseReranker(Reranker):

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        sorting_method: str,
        **kwargs: dict[str, Any],
    ):
        return getattr(self, f"_{sorting_method}")(query, candidates, ranking_func, **kwargs)

    def _all_pair(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        **kwargs: dict[str, Any],
    ) -> tuple[list[str], list[int]]:
        doc_pairs = list(combinations(range(len(candidates)), 2))
        scores = [0] * len(candidates)
        for i, j in doc_pairs:
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
        return ranked_result, ranked_indices

    def _bubble_sort(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        topk: int = 10,
        **kwargs: dict[str, Any],
    ) -> tuple[list[str], list[int]]:
        last_end = len(candidates) - 1
        ranked_indices = list(range(len(candidates)))
        for i in range(min(topk, len(candidates))):
            changed = False
            for j in range(last_end, i, -1):
                if ranking_func(query, candidates[j], candidates[j - 1], **kwargs) > 0:
                    candidates[j - 1], candidates[j] = candidates[j], candidates[j - 1]
                    ranked_indices[j - 1], ranked_indices[j] = ranked_indices[j], ranked_indices[j - 1]
                    if not changed:
                        changed = True
                        # skip unchanged pairs at the bottom
                        if last_end != len(candidates) - 1:
                            last_end += 1
                if not changed:
                    last_end -= 1
        return candidates, ranked_indices

    def _heap_sort(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        topk: int = 10,
        **kwargs: dict[str, Any],
    ) -> tuple[list[str], list[int]]:

        ranked_indices = list(range(len(candidates)))

        def heapify(n, i):
            # Find largest among root and children
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            if l < n and ranking_func(query, candidates[l], candidates[i], **kwargs) > 0:
                largest = l
            if r < n and ranking_func(query, candidates[r], candidates[largest], **kwargs) > 0:
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

        return reversed(candidates), reversed(ranked_indices)


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
        **kwargs: dict[str, Any],
    ) -> tuple[list[str], list[int]]:

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
            permutation = ranking_func(query, ranked_result[start_pos:end_pos], **kwargs)

            # receive permutation
            cut_range = copy.deepcopy(ranked_result[start_pos:end_pos])
            cut_range_indices = copy.deepcopy(ranked_indices[start_pos:end_pos])
            for local_rank, index in enumerate(permutation):
                ranked_result[start_pos + local_rank] = copy.deepcopy(cut_range[index])
                ranked_indices[start_pos + local_rank] = cut_range_indices[index]

            start_pos, end_pos = start_pos - step, end_pos - step

        return ranked_result, ranked_indices


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
        **kwargs: dict[str, Any],
    ) -> tuple[list[str], list[int]]:
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

                    top_indices = ranking_func(query, group_candidates, promotion_size, **kwargs)  # select top M form N candidates
                    top_ids = [group_candidate_ids[i] for i in top_indices]  # convert to global index
                    stage_ids.extend(top_ids)

                    for doc_id in top_ids:
                        doc_scores[doc_id] += 1

                stage_ids = sorted(stage_ids, key=lambda x: doc_scores[x], reverse=True)

        ranked_indices, _ = zip(*sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True))
        ranked_result = [candidates[i] for i in ranked_indices]
        return ranked_result, ranked_indices
