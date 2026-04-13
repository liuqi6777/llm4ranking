import copy
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional

from llm4ranking.policy.base import (
    BaseRankingPolicy,
    ListwisePolicy,
    PairwisePolicy,
    PointwisePolicy,
    SelectionPolicy,
)


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


@dataclass
class RerankResult:
    documents: list[str]
    indices: list[int]
    record: Optional[RankingRecord] = None


def _chunk_sequence(items: list[Any], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


class RerankStrategy(ABC):
    compatible_model_kind = "base"

    def validate_ranking_model(self, ranking_func: BaseRankingPolicy) -> None:
        model_kind = getattr(ranking_func, "model_kind", "base")
        if model_kind != self.compatible_model_kind:
            raise ValueError(
                f"{self.__class__.__name__} expects a {self.compatible_model_kind} model, got {model_kind}."
            )

    def _create_record(
        self,
        *,
        query: str,
        candidates: list[str],
        ranking_func: BaseRankingPolicy,
        return_record: bool,
        num_processed_docs: int = 0,
    ) -> tuple[Optional[RankingRecord], Optional[float]]:
        if not return_record:
            return None, None

        return (
            RankingRecord(
                query=query,
                candidates=list(candidates),
                num_processed_docs=num_processed_docs,
                prompt_template=ranking_func._prompt_template,
            ),
            time.time(),
        )

    def _finalize_record(
        self,
        *,
        record: Optional[RankingRecord],
        started_at: Optional[float],
        rank_indices: list[int],
    ) -> None:
        if record is None or started_at is None:
            return

        record.latency = time.time() - started_at
        record.rank_indices = rank_indices

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: BaseRankingPolicy,
        return_record: bool = False,
        **kwargs: dict[str, Any],
    ) -> RerankResult:
        pass


class Pointwise(RerankStrategy):
    compatible_model_kind = "pointwise"

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: PointwisePolicy,
        truncate_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_record: bool = False,
        **kwargs: dict[str, Any],
    ) -> RerankResult:
        self.validate_ranking_model(ranking_func)

        candidates_for_scoring = list(candidates)
        record, started_at = self._create_record(
            query=query,
            candidates=candidates,
            ranking_func=ranking_func,
            return_record=return_record,
            num_processed_docs=len(candidates),
        )

        if truncate_length:
            candidates_for_scoring = [
                " ".join(candidate.split()[:truncate_length]) for candidate in candidates_for_scoring
            ]

        scores = []
        effective_batch_size = len(candidates_for_scoring) if batch_size is None else max(batch_size, 1)
        use_batch = (
            len(candidates_for_scoring) > 0
            and getattr(ranking_func, "supports_batch", False)
        )
        if use_batch:
            for candidate_batch in _chunk_sequence(candidates_for_scoring, effective_batch_size):
                if return_record:
                    batch_result = ranking_func.score_many(
                        query,
                        candidate_batch,
                        return_lm_outputs=True,
                        **kwargs,
                    )
                    record.lm_outputs.append(batch_result.lm_outputs)
                    record.num_lm_calls += 1
                    record.batch_sizes.append(len(candidate_batch))
                    batch_scores = batch_result.value
                else:
                    batch_scores = ranking_func.score_many(query, candidate_batch, **kwargs)
                scores.extend(batch_scores)
        else:
            for candidate in candidates_for_scoring:
                if return_record:
                    result = ranking_func.score(query, candidate, return_lm_outputs=True, **kwargs)
                    record.lm_outputs.append(result.lm_outputs)
                    record.num_lm_calls += 1
                    record.batch_sizes.append(1)
                    score = result.value
                else:
                    score = ranking_func.score(query, candidate, **kwargs)
                scores.append(score)

        if len(scores) == 0:
            ranked_indices = []
            ranked_result = []
        else:
            ranked_indices, _ = zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
            ranked_indices = list(ranked_indices)
            ranked_result = [candidates[i] for i in ranked_indices]

        self._finalize_record(record=record, started_at=started_at, rank_indices=ranked_indices)

        return RerankResult(
            documents=ranked_result,
            indices=ranked_indices,
            record=record,
        )


class Pairwise(RerankStrategy):
    compatible_model_kind = "pairwise"

    def _record_pairwise_invocation(
        self,
        record: RankingRecord,
        *,
        lm_outputs: Any,
        batch_size: int,
    ) -> None:
        record.num_processed_docs += 4 * batch_size
        record.lm_outputs.append(lm_outputs)
        record.num_lm_calls += 2
        record.batch_sizes.extend([batch_size, batch_size])

    def _compare_pair(
        self,
        *,
        query: str,
        left_doc: str,
        right_doc: str,
        ranking_func: PairwisePolicy,
        return_record: bool,
        record: Optional[RankingRecord],
        **kwargs,
    ):
        if return_record:
            result = ranking_func.compare(
                query,
                left_doc,
                right_doc,
                return_lm_outputs=True,
                **kwargs,
            )
            self._record_pairwise_invocation(record, lm_outputs=result.lm_outputs, batch_size=1)
            return result.value
        return ranking_func.compare(query, left_doc, right_doc, **kwargs)


class PairwiseAllPair(Pairwise):

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: PairwisePolicy,
        pair_batch_size: Optional[int] = None,
        return_record: bool = False,
        **kwargs: dict[str, Any],
    ) -> RerankResult:
        self.validate_ranking_model(ranking_func)

        record, started_at = self._create_record(
            query=query,
            candidates=candidates,
            ranking_func=ranking_func,
            return_record=return_record,
            num_processed_docs=len(candidates),
        )

        doc_pairs = list(combinations(range(len(candidates)), 2))
        scores = [0] * len(candidates)
        effective_batch_size = len(doc_pairs) if pair_batch_size is None else max(pair_batch_size, 1)
        use_batch = (
            len(doc_pairs) > 0
            and getattr(ranking_func, "supports_batch", False)
        )
        if use_batch:
            for pair_batch in _chunk_sequence(doc_pairs, effective_batch_size):
                text_pairs = [(candidates[i], candidates[j]) for i, j in pair_batch]
                if return_record:
                    batch_result = ranking_func.compare_many(
                        query,
                        text_pairs,
                        return_lm_outputs=True,
                        **kwargs,
                    )
                    self._record_pairwise_invocation(
                        record,
                        lm_outputs=batch_result.lm_outputs,
                        batch_size=len(pair_batch),
                    )
                    batch_scores = batch_result.value
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
                    result = ranking_func.compare(
                        query,
                        candidates[i],
                        candidates[j],
                        return_lm_outputs=True,
                        **kwargs,
                    )
                    self._record_pairwise_invocation(record, lm_outputs=result.lm_outputs, batch_size=1)
                    res = result.value
                else:
                    res = ranking_func.compare(query, candidates[i], candidates[j], **kwargs)

                if res > 0:
                    scores[i] += 1
                elif res < 0:
                    scores[j] += 1
                else:
                    scores[i] += 0.5
                    scores[j] += 0.5

        if len(scores) == 0:
            ranked_indices = []
            ranked_result = []
        else:
            ranked_indices, _ = zip(*sorted(enumerate(scores), key=lambda x: x[1], reverse=True))
            ranked_indices = list(ranked_indices)
            ranked_result = [candidates[i] for i in ranked_indices]

        self._finalize_record(record=record, started_at=started_at, rank_indices=ranked_indices)

        return RerankResult(
            documents=ranked_result,
            indices=ranked_indices,
            record=record,
        )


class PairwiseBubbleSort(Pairwise):

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: PairwisePolicy,
        topk: int = 10,
        return_record: bool = False,
        **kwargs: dict[str, Any],
    ) -> RerankResult:
        self.validate_ranking_model(ranking_func)

        ranked_docs = list(candidates)
        ranked_indices = list(range(len(candidates)))
        record, started_at = self._create_record(
            query=query,
            candidates=candidates,
            ranking_func=ranking_func,
            return_record=return_record,
            num_processed_docs=len(candidates),
        )

        last_end = len(ranked_docs) - 1
        for i in range(min(topk, len(ranked_docs))):
            changed = False
            for j in range(last_end, i, -1):
                res = self._compare_pair(
                    query=query,
                    left_doc=ranked_docs[j],
                    right_doc=ranked_docs[j - 1],
                    ranking_func=ranking_func,
                    return_record=return_record,
                    record=record,
                    **kwargs,
                )

                if res > 0:
                    ranked_docs[j - 1], ranked_docs[j] = ranked_docs[j], ranked_docs[j - 1]
                    ranked_indices[j - 1], ranked_indices[j] = ranked_indices[j], ranked_indices[j - 1]
                    if not changed:
                        changed = True
                        if last_end != len(ranked_docs) - 1:
                            last_end += 1
                if not changed:
                    last_end -= 1

        self._finalize_record(record=record, started_at=started_at, rank_indices=ranked_indices)

        return RerankResult(
            documents=ranked_docs,
            indices=ranked_indices,
            record=record,
        )


class PairwiseHeapSort(Pairwise):

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: PairwisePolicy,
        topk: int = 10,
        return_record: bool = False,
        **kwargs: dict[str, Any],
    ) -> RerankResult:
        self.validate_ranking_model(ranking_func)

        heap_docs = list(candidates)
        ranked_indices = list(range(len(candidates)))
        record, started_at = self._create_record(
            query=query,
            candidates=candidates,
            ranking_func=ranking_func,
            return_record=return_record,
            num_processed_docs=0,
        )

        def wrapped_ranking_func(i, j):
            return self._compare_pair(
                query=query,
                left_doc=heap_docs[i],
                right_doc=heap_docs[j],
                ranking_func=ranking_func,
                return_record=return_record,
                record=record,
                **kwargs,
            )

        def heapify(n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and wrapped_ranking_func(left, i) > 0:
                largest = left
            if right < n and wrapped_ranking_func(right, largest) > 0:
                largest = right
            if largest != i:
                heap_docs[i], heap_docs[largest] = heap_docs[largest], heap_docs[i]
                ranked_indices[i], ranked_indices[largest] = ranked_indices[largest], ranked_indices[i]
                heapify(n, largest)

        n = len(heap_docs)
        for i in range(n // 2, -1, -1):
            heapify(n, i)
        for i in range(n - 1, max(n - 1 - topk, 0), -1):
            heap_docs[i], heap_docs[0] = heap_docs[0], heap_docs[i]
            ranked_indices[i], ranked_indices[0] = ranked_indices[0], ranked_indices[i]
            heapify(i, 0)

        ranked_docs = list(reversed(heap_docs))
        ranked_ids = list(reversed(ranked_indices))
        self._finalize_record(record=record, started_at=started_at, rank_indices=ranked_ids)

        return RerankResult(
            documents=ranked_docs,
            indices=ranked_ids,
            record=record,
        )


class ListwiseSlidingWindow(RerankStrategy):
    compatible_model_kind = "listwise"

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: ListwisePolicy,
        rank_start: int = 0,
        rank_end: int = None,
        window_size: Optional[int] = None,
        step: int = 10,
        truncate_length: Optional[int] = None,
        return_record: bool = False,
        **kwargs: dict[str, Any],
    ) -> RerankResult:
        self.validate_ranking_model(ranking_func)

        record, started_at = self._create_record(
            query=query,
            candidates=candidates,
            ranking_func=ranking_func,
            return_record=return_record,
            num_processed_docs=0,
        )

        ranked_result = list(candidates)
        ranked_inputs = list(candidates)
        if truncate_length:
            ranked_inputs = [" ".join(candidate.split()[:truncate_length]) for candidate in ranked_inputs]
        ranked_indices = list(range(len(candidates)))

        window_size = window_size or len(candidates)
        rank_end = rank_end or len(candidates)
        start_pos, end_pos = rank_end - window_size, rank_end
        while end_pos > rank_start and start_pos + step != rank_start:
            start_pos = max(start_pos, rank_start)

            if return_record:
                result = ranking_func.rank(
                    query,
                    ranked_inputs[start_pos:end_pos],
                    return_lm_outputs=True,
                    **kwargs,
                )
                record.num_processed_docs += end_pos - start_pos
                record.lm_outputs.append(result.lm_outputs)
                permutation = result.value
            else:
                permutation = ranking_func.rank(query, ranked_inputs[start_pos:end_pos], **kwargs)

            cut_range = copy.deepcopy(ranked_result[start_pos:end_pos])
            cut_range_inputs = copy.deepcopy(ranked_inputs[start_pos:end_pos])
            cut_range_indices = copy.deepcopy(ranked_indices[start_pos:end_pos])
            for local_rank, index in enumerate(permutation):
                ranked_result[start_pos + local_rank] = copy.deepcopy(cut_range[index])
                ranked_inputs[start_pos + local_rank] = copy.deepcopy(cut_range_inputs[index])
                ranked_indices[start_pos + local_rank] = cut_range_indices[index]

            start_pos, end_pos = start_pos - step, end_pos - step

        self._finalize_record(record=record, started_at=started_at, rank_indices=ranked_indices)

        return RerankResult(
            documents=ranked_result,
            indices=ranked_indices,
            record=record,
        )


class Tournament(RerankStrategy):
    compatible_model_kind = "selection"

    def rerank(
        self,
        query: str,
        candidates: list[str],
        ranking_func: SelectionPolicy,
        tournament_times: int = 1,
        group_sizes: tuple[int] = (10, 10, 10, 10, 5),
        promotion_sizes: tuple[int] = (5, 4, 5, 5, 2),
        stage_weight: float = 0.0,
        truncate_length: Optional[int] = None,
        return_record: bool = False,
        **kwargs: dict[str, Any],
    ) -> RerankResult:
        self.validate_ranking_model(ranking_func)

        record, started_at = self._create_record(
            query=query,
            candidates=candidates,
            ranking_func=ranking_func,
            return_record=return_record,
            num_processed_docs=0,
        )

        if len(group_sizes) != len(promotion_sizes):
            raise ValueError("group_sizes and promotion_sizes must have the same length.")

        model_candidates = list(candidates)
        if truncate_length:
            model_candidates = [" ".join(candidate.split()[:truncate_length]) for candidate in model_candidates]

        if len(candidates) == 0:
            return RerankResult(documents=[], indices=[], record=record)

        doc_scores = [0] * len(candidates)

        for _ in range(tournament_times):
            stage_ids = list(range(len(candidates)))

            for stage, (group_size, promotion_size) in enumerate(zip(group_sizes, promotion_sizes)):
                num_groups = max(1, math.ceil(len(stage_ids) / group_size))
                groups = [[] for _ in range(num_groups)]
                for idx, value in enumerate(stage_ids):
                    groups[idx % num_groups].append(value)

                new_stage = []
                for group_candidate_ids in groups:
                    if not group_candidate_ids:
                        continue

                    random.shuffle(group_candidate_ids)
                    group_candidates = [model_candidates[i] for i in group_candidate_ids]

                    if return_record:
                        result = ranking_func.select(
                            query,
                            group_candidates,
                            min(promotion_size, len(group_candidates)),
                            return_lm_outputs=True,
                            **kwargs,
                        )
                        record.num_processed_docs += len(group_candidates)
                        record.lm_outputs.append(result.lm_outputs)
                        top_indices = result.value
                    else:
                        top_indices = ranking_func.select(
                            query,
                            group_candidates,
                            min(promotion_size, len(group_candidates)),
                            **kwargs,
                        )

                    global_ids = [group_candidate_ids[i] for i in top_indices]
                    new_stage.extend(global_ids)

                    for gid in global_ids:
                        doc_scores[gid] += 1 + stage * stage_weight

                stage_ids = sorted(new_stage, key=lambda x: doc_scores[x], reverse=True)
                if len(stage_ids) <= 1:
                    break

        ranked_indices, _ = zip(*sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True))
        ranked_indices = list(ranked_indices)
        ranked_result = [candidates[i] for i in ranked_indices]

        self._finalize_record(record=record, started_at=started_at, rank_indices=ranked_indices)

        return RerankResult(
            documents=ranked_result,
            indices=ranked_indices,
            record=record,
        )
