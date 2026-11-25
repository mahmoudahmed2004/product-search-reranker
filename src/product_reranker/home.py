"""Topic selection from keyword and semantic retrieval evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DENSE_CONFIDENCE = 0.48
DENSE_MARGIN = 0.04


@dataclass(frozen=True, slots=True)
class Hit:
    id: int
    collection: str
    score: float
    kind: str = ""


def select_topic(
    dense: Sequence[Hit], sparse: Sequence[Hit], exact_collection: str | None = None
) -> str | None:
    if exact_collection:
        return exact_collection
    if not dense:
        return None
    first = dense[0]
    consensus = sum(hit.collection == first.collection for hit in dense[:3])
    rival = next((hit.score for hit in dense if hit.collection != first.collection), 0.0)
    if first.score >= 0.8:
        if sparse and first.id == sparse[0].id and first.score - rival >= 0.025:
            return first.collection
        if (
            not sparse
            and len(dense) >= 3
            and first.kind
            and dense[1].collection == first.collection
            and first.kind == dense[1].kind
            and first.score - dense[2].score >= 0.02
        ):
            return first.collection
    if consensus < 2:
        return None
    if first.score >= 0.27 and sparse and sparse[0].collection == first.collection:
        return first.collection
    if first.score >= DENSE_CONFIDENCE and first.score - rival >= DENSE_MARGIN:
        return first.collection
    return None


def rank_topic_hits(topic: str, dense: Sequence[Hit], sparse: Sequence[Hit]) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    winner = next((hit.score for hit in dense if hit.collection == topic), 0.0)
    cutoff = max(winner - 0.12, 0.26)
    for rank, hit in enumerate(dense, 1):
        if hit.collection == topic and hit.score >= cutoff:
            scores[hit.id] += 1 / (8 + rank)
    for rank, hit in enumerate(sparse, 1):
        if hit.collection == topic:
            scores[hit.id] += 1 / (8 + rank)
    return sorted(scores, key=lambda item_id: (-scores[item_id], item_id))


def preserve_query_evidence(
    retrieval: Sequence[int], learned: list[int], evidence: Mapping[int, float]
) -> list[int]:
    """Keep a materially stronger match ahead of a learned promotion."""
    if retrieval and learned:
        first = retrieval[0]
        if evidence[learned[0]] < 0.75 * evidence[first]:
            return [first, *(item_id for item_id in learned if item_id != first)]
    return learned
