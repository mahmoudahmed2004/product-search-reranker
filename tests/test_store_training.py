import pytest

pytest.importorskip("fastembed")

from product_reranker.core import Candidate  # noqa: E402
from product_reranker.relations import PairProduct, pair_features  # noqa: E402
from product_reranker.train_store import PairGroup, disjoint_anchors  # noqa: E402


def pair(anchor, *targets):
    return PairGroup(
        Candidate(anchor, anchor),
        tuple(Candidate(target, target) for target in targets),
        tuple("C" for target in targets),
    )


def test_split_isolates_anchor_ids_and_reverse_pairs():
    train, validation, test = disjoint_anchors(
        [pair("a", "b", "c"), pair("b", "x"), pair("z", "y")],
        [pair("a", "v"), pair("d", "e")],
        [pair("b", "a")],
    )
    assert {group.anchor.product_id for group in validation} == {"a", "d"}
    assert {group.anchor.product_id for group in train} == {"z"}
    assert [group.anchor.product_id for group in test] == ["b"]
    train, _, _ = disjoint_anchors([pair("a", "b", "c")], [], [pair("b", "a")])
    assert [target.product_id for target in train[0].candidates] == ["c"]


def test_pair_features_include_model_digits_and_accessory_cues():
    anchor = PairProduct("iPhone 13 Pro", "", "Apple")
    close = PairProduct("iPhone 13 Pro Max", "", "Apple")
    accessory = PairProduct("Protective case for iPhone 12", "", "Other")
    matrix = pair_features(anchor, [close, accessory])
    assert matrix[0, 0] > matrix[1, 0]
    assert matrix[:, 4].tolist() == [1, 0]
    assert matrix[:, 7].tolist() == [0, 1]
