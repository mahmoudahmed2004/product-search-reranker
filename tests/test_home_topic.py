import pytest

from product_reranker.home import Hit, preserve_query_evidence, select_topic


def test_exact_collection_wins_over_empty_and_conflicting_signals() -> None:
    assert select_topic([], [], exact_collection="espresso_machines") == "espresso_machines"

    dense = [
        Hit(1, "running_shoes", 0.88),
        Hit(2, "running_shoes", 0.85),
        Hit(3, "sandals", 0.40),
    ]
    sparse = [Hit(1, "running_shoes", 9.5), Hit(2, "running_shoes", 8.0)]
    assert select_topic(dense, sparse, exact_collection="espresso_machines") == "espresso_machines"


@pytest.mark.parametrize(
    "dense",
    [
        [
            Hit(1, "hiking_boots", 0.62),
            Hit(2, "hiking_boots", 0.56),
            Hit(3, "sandals", 0.45),
        ],
        [
            Hit(1, "hiking_boots", 0.48),
            Hit(2, "hiking_boots", 0.47),
            Hit(3, "sandals", 0.41),
        ],
        [
            Hit(1, "hiking_boots", 0.66),
            Hit(2, "sandals", 0.54),
            Hit(3, "hiking_boots", 0.53),
        ],
    ],
)
def test_dense_only_qualifies_when_thresholds_and_consensus_met(dense: list[Hit]) -> None:
    assert select_topic(dense, []) == "hiking_boots"


@pytest.mark.parametrize(
    "dense",
    [
        # Cosine below 0.48 despite consensus and margin
        [
            Hit(1, "hiking_boots", 0.47),
            Hit(2, "hiking_boots", 0.45),
            Hit(3, "sandals", 0.35),
        ],
        # Fewer than two of top three in chosen collection (4th hit does not count)
        [
            Hit(1, "hiking_boots", 0.65),
            Hit(2, "sandals", 0.55),
            Hit(3, "sneakers", 0.52),
            Hit(4, "hiking_boots", 0.51),
        ],
        # Cross-collection margin below 0.04
        [
            Hit(1, "hiking_boots", 0.60),
            Hit(2, "hiking_boots", 0.59),
            Hit(3, "sandals", 0.58),
        ],
    ],
)
def test_dense_only_rejects_when_any_gate_fails(dense: list[Hit]) -> None:
    assert select_topic(dense, []) is None


def test_sparse_corroboration_qualifies_with_modest_dense_scores() -> None:
    dense = (
        Hit(1, "coffee_grinders", 0.44),
        Hit(2, "coffee_grinders", 0.42),
        Hit(3, "mugs", 0.35),
    )
    sparse = (
        Hit(1, "coffee_grinders", 6.5),
        Hit(2, "coffee_grinders", 4.2),
        Hit(3, "mugs", 1.1),
    )
    assert select_topic(dense, ()) is None
    assert select_topic(dense, sparse) == "coffee_grinders"


@pytest.mark.parametrize(
    ("dense", "sparse"),
    [
        (
            [
                Hit(1, "blenders", 0.52),
                Hit(2, "food_processors", 0.50),
                Hit(3, "blenders", 0.49),
            ],
            [
                Hit(10, "food_processors", 5.4),
                Hit(11, "blenders", 5.1),
            ],
        ),
        (
            [
                Hit(1, "running_shoes", 0.44),
                Hit(2, "running_shoes", 0.42),
                Hit(3, "trail_shoes", 0.40),
            ],
            [
                Hit(10, "trail_shoes", 5.0),
                Hit(11, "running_shoes", 4.7),
            ],
        ),
        (
            [
                Hit(1, "mugs", 0.42),
                Hit(2, "kettles", 0.40),
                Hit(3, "spoons", 0.39),
            ],
            [
                Hit(10, "coffee_grinders", 5.5),
                Hit(11, "mugs", 5.0),
            ],
        ),
    ],
)
def test_sparse_dense_disagreement_at_close_scores_returns_none(
    dense: list[Hit], sparse: list[Hit]
) -> None:
    assert select_topic(dense, sparse) is None


@pytest.mark.parametrize(
    ("dense", "sparse"),
    [
        ([], []),
        ([], [Hit(1, "mugs", 2.0), Hit(2, "plates", 1.8)]),
        ([Hit(1, "mugs", 0.78)], []),
        (
            [Hit(1, "a", 0.21), Hit(2, "b", 0.19), Hit(3, "c", 0.15)],
            [Hit(4, "d", 0.6), Hit(5, "e", 0.5)],
        ),
        ([Hit(1, "mugs", 0.18), Hit(2, "mugs", 0.15), Hit(3, "mugs", 0.12)], []),
    ],
)
def test_empty_and_noisy_inputs_return_none(dense: list[Hit], sparse: list[Hit]) -> None:
    assert select_topic(dense, sparse) is None


def test_one_strong_lexical_and_semantic_match_handles_small_departments() -> None:
    dense = [Hit(1, "pets", 0.83), Hit(2, "pantry", 0.78), Hit(3, "kitchen", 0.77)]
    assert select_topic(dense, [Hit(1, "pets", 5.0)]) == "pets"
    assert select_topic(dense, [Hit(2, "pantry", 5.0)]) is None


def test_semantic_results_must_converge_on_a_specific_kind() -> None:
    dense = [
        Hit(1, "kitchen", 0.81, "strainer"),
        Hit(2, "kitchen", 0.80, "strainer"),
        Hit(3, "pantry", 0.78, "rice"),
    ]
    assert select_topic(dense, []) == "kitchen"
    dense[1] = Hit(2, "kitchen", 0.80, "blender")
    assert select_topic(dense, []) is None


def test_reranker_respects_strong_query_evidence_but_can_correct_a_tie() -> None:
    assert preserve_query_evidence([1, 2, 3], [2, 3, 1], {1: 2.0, 2: 1.0}) == [1, 2, 3]
    assert preserve_query_evidence([1, 2], [2, 1], {1: 1.0, 2: 1.0}) == [2, 1]
