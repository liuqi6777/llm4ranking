import copy
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
    ) -> list[str]:
        pass


class PointwiseReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str], float],
        **kwargs: dict[str, Any],
    ) -> list[str]:
        return sorted(candidates, key=lambda x: ranking_func(query, x), reverse=True)


class PairwiseAllPairReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        **kwargs: dict[str, Any],
    ) -> list[str]:
        doc_pairs = list(combinations(range(len(candidates)), 2))
        scores = [0] * len(candidates)
        for i, j in doc_pairs:
            res = ranking_func(query, candidates[i], candidates[j])
            if res > 0:
                scores[i] += 1
            elif res < 0:
                scores[j] += 1
            else:
                scores[i] += 0.5
                scores[j] += 0.5
        return [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]


class PairwiseBubbleReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        topk: int = 10,
        **kwargs: dict[str, Any],
    ) -> list[str]:
        last_end = len(candidates) - 1
        for i in range(min(topk, len(candidates))):
            changed = False
            for j in range(last_end, i, -1):
                if ranking_func(query, candidates[j], candidates[j - 1]) > 0:
                    candidates[j - 1], candidates[j] = candidates[j], candidates[j - 1]
                    if not changed:
                        changed = True
                        if last_end != len(candidates) - 1:  # skip unchanged pairs at the bottom
                            last_end += 1
                if not changed:
                    last_end -= 1
        return candidates


class PairwiseHeapReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, str, str], int],
        topk: int = 10,
        **kwargs: dict[str, Any],
    ) -> list[str]:

        def heapify(n, i):
            # Find largest among root and children
            largest = i
            l = 2 * i + 1
            r = 2 * i + 2
            if l < n and ranking_func(query, candidates[l], candidates[i]) > 0:
                largest = l
            if r < n and ranking_func(query, candidates[r], candidates[largest]) > 0:
                largest = r
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                candidates[i], candidates[largest] = candidates[largest], candidates[i]
                heapify(n, largest)

        n = len(candidates)
        # Build max heap
        for i in range(n // 2, -1, -1):
            heapify(n, i)
        for i in range(n - 1, max(n - 1 - topk, 0), -1):
            candidates[i], candidates[0] = candidates[0], candidates[i]
            # Heapify root element
            heapify(i, 0)

        return candidates


class ListwiseReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: Callable[[str, list[str]], list[int]],
        **kwargs: dict[str, Any],
    ) -> list[str]:

        return [doc for _, doc in sorted(zip(ranking_func(query, candidates), candidates))]
        

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
        **kwargs: dict[str, Any],
    ) -> list[str]:

        rerank_result = copy.deepcopy(candidates)

        window_size = window_size or len(candidates)
        rank_end = rank_end or len(candidates)
        start_pos, end_pos = rank_end - window_size, rank_end
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)
            permutation = ranking_func(query, rerank_result[start_pos:end_pos])  # range from 0 to window_size

            # receive permutation
            cut_range = copy.deepcopy(rerank_result[start_pos:end_pos])
            for local_rank, index in enumerate(permutation):
                rerank_result[start_pos + local_rank] = copy.deepcopy(cut_range[index])

            start_pos, end_pos = start_pos - step, end_pos - step

        return rerank_result
