import math

import pytest

from product_reranker.core import Candidate, Featurizer, normalize_query, rank_indices
from product_reranker.dataset import QueryGroup, split_groups
from product_reranker.evaluate import ndcg_at_10
from product_reranker.model import Ranker, train_ranker


def group(query_id, query, split):
    return QueryGroup(
        query_id,
        query,
        (Candidate("a", f"{query} replacement case"), Candidate("b", query)),
        ("C", "E"),
        split,
    )


def test_ndcg_uses_esci_gains_and_cutoff():
    assert ndcg_at_10(["S", "E"]) == pytest.approx(
        (10 + 100 / math.log2(3)) / (100 + 10 / math.log2(3))
    )
    assert ndcg_at_10(["I"] * 10 + ["E"]) == 0
    assert ndcg_at_10(["I", "I"]) == 0


def test_normalized_query_stays_in_one_split_and_official_test_takes_precedence():
    assert normalize_query("  Ｐhone-case! ") == "phone case"
    groups = [group("leak", "PHONE-case!", "train"), group("held", "Phone case", "test")]
    groups.extend(group(str(i), f"desk lamp {i}", "train") for i in range(60))
    split = split_groups(groups)
    texts = [
        {normalize_query(g.query) for g in part}
        for part in (split.train, split.validation, split.test)
    ]
    assert all(texts[i].isdisjoint(texts[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert split.overlap_dropped == 1
    assert split.test[0].query_id == "held"
    assert len(split.test[0].candidates) == 2


def test_bm25_uses_title_and_description_and_ties_are_deterministic():
    candidates = (
        Candidate("z", "Black cup"),
        Candidate("a", "Blue cup"),
        Candidate("b", "Travel set", "Contains a steel bottle."),
    )
    features = Featurizer.fit(candidates)
    rows = features.transform("steel bottle", candidates)
    assert rank_indices(rows[:, 0], candidates) == [2, 1, 0]
    assert rank_indices([3, 3, 3], candidates) == [1, 2, 0]


def test_model_round_trip_keeps_the_same_candidates(tmp_path):
    train = [group(f"train-{i}", f"steel bottle {i}", "train") for i in range(25)]
    valid = [group("valid", "stainless mug", "train")]
    ranker = train_ranker(train, valid)
    ranker.save(tmp_path)
    restored = Ranker.load(tmp_path)
    trial = valid[0]
    result = restored.rank(trial.query, trial.candidates)
    assert result == ranker.rank(trial.query, trial.candidates)
    assert {row.product_id for row in result} == {"a", "b"}
    assert {row.bm25_rank for row in result} == {1, 2}
    assert {row.lambdarank_rank for row in result} == {1, 2}
