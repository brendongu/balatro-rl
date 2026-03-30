"""Tests for feature extraction modules."""

from balatro_rl.features.hand_evaluator import (
    HandCandidate,
    best_play,
    detect_hand_type,
    evaluate_hand,
    parse_hand,
)


def _card(cid: int, key: str, suit: str, rank: str) -> dict:
    return {
        "id": cid, "key": key,
        "value": {"suit": suit, "rank": rank},
        "modifier": {}, "state": {"debuff": False},
    }


SAMPLE_HAND = [
    _card(1, "H_A", "H", "A"),
    _card(2, "H_K", "H", "K"),
    _card(3, "H_Q", "H", "Q"),
    _card(4, "H_J", "H", "J"),
    _card(5, "H_T", "H", "T"),
    _card(6, "D_9", "D", "9"),
    _card(7, "C_8", "C", "8"),
    _card(8, "S_7", "S", "7"),
]


def test_evaluate_hand_returns_candidates():
    candidates = evaluate_hand(SAMPLE_HAND)
    assert len(candidates) > 0
    assert all(isinstance(c, HandCandidate) for c in candidates)


def test_best_play_is_straight_flush():
    bp = best_play(SAMPLE_HAND)
    assert bp.hand_type == "Straight Flush"
    assert len(bp.scoring_cards) == 5


def test_candidates_sorted_by_score():
    candidates = evaluate_hand(SAMPLE_HAND)
    for i in range(len(candidates) - 1):
        assert candidates[i].estimated_chips >= candidates[i + 1].estimated_chips


def test_parse_hand_length():
    parsed = parse_hand(SAMPLE_HAND)
    assert len(parsed) == 8


def test_detect_pair():
    pair_hand = [
        _card(0, "H_A", "H", "A"),
        _card(1, "D_A", "D", "A"),
    ]
    parsed = parse_hand(pair_hand)
    hand_type, scoring = detect_hand_type(parsed)
    assert hand_type == "Pair"
    assert len(scoring) == 2
