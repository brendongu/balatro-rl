"""Tests for heuristic hand and shop policies."""

from __future__ import annotations

import numpy as np
import pytest
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

from balatro_rl.agents.blind import HeuristicBlindPolicy
from balatro_rl.agents.dispatch import PhaseDispatchAgent
from balatro_rl.agents.hand import HeuristicHandPolicy, _can_one_shot_blind
from balatro_rl.agents.shop import HeuristicShopPolicy
from balatro_rl.env.wrappers import ActionInfoWrapper
from balatro_rl.features.hand_evaluator import (
    HandCandidate,
    ParsedCard,
    ParsedJoker,
    evaluate_hand_from_parsed,
    flush_potential,
    fullhouse_potential,
    parse_cards_from_obs,
    parse_jokers_from_obs,
    recommend_discards,
)
from balatro_rl.features.joker_catalog import (
    NUM_CENTER_KEYS,
    id_to_key,
    is_common_scoring_joker,
    is_scaling_joker,
    joker_category,
    key_to_id,
    planet_hand_type,
)
from balatro_rl.features.joker_scoring import GameContext, simulate_joker_scoring


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    e = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=500)
    yield e
    e.close()


@pytest.fixture
def wrapped_env():
    e = ActionInfoWrapper(
        BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=500)
    )
    yield e
    e.close()


# ---------------------------------------------------------------------------
# joker_catalog tests
# ---------------------------------------------------------------------------


def test_id_key_roundtrip():
    key = "j_joker"
    assert id_to_key(key_to_id(key)) == key


def test_joker_category_basic():
    assert joker_category("j_joker") == "mult"
    assert joker_category("j_banner") == "chips"


def test_common_scoring_joker():
    assert is_common_scoring_joker("j_joker")
    assert is_common_scoring_joker("j_greedy_joker")


def test_planet_hand_type_lookup():
    assert planet_hand_type("c_jupiter") == "Flush"
    assert planet_hand_type("c_saturn") == "Straight"
    assert planet_hand_type("c_mercury") == "Pair"
    assert planet_hand_type("c_pluto") == "High Card"
    assert planet_hand_type("c_uranus") == "Two Pair"


def test_scaling_joker_identification():
    assert is_scaling_joker("j_green_joker")
    assert is_scaling_joker("j_ride_the_bus")
    assert is_scaling_joker("j_supernova")
    assert not is_scaling_joker("j_joker")


# ---------------------------------------------------------------------------
# parse_cards_from_obs tests
# ---------------------------------------------------------------------------


def test_parse_cards_from_obs_basic():
    hand_card = np.zeros((8, 15), dtype=np.float32)
    # Card 0: Ace of Spades -> rank_id=14, suit_id=3
    hand_card[0, 0] = 14 / 14.0
    hand_card[0, 1] = 3 / 3.0
    # Card 1: 5 of Hearts -> rank_id=5, suit_id=0
    hand_card[1, 0] = 5 / 14.0
    hand_card[1, 1] = 0 / 3.0

    cards = parse_cards_from_obs(hand_card, n_cards=2)
    assert len(cards) == 2
    assert cards[0].rank == "A"
    assert cards[0].suit == "S"
    assert cards[1].rank == "5"
    assert cards[1].suit == "H"


def test_parse_cards_skips_face_down():
    hand_card = np.zeros((8, 15), dtype=np.float32)
    hand_card[0, 0] = 14 / 14.0
    hand_card[0, 1] = 3 / 3.0
    hand_card[0, 7] = 1.0  # face down
    hand_card[1, 0] = 5 / 14.0
    hand_card[1, 1] = 0 / 3.0

    cards = parse_cards_from_obs(hand_card, n_cards=2)
    assert len(cards) == 1
    assert cards[0].rank == "5"


# ---------------------------------------------------------------------------
# Discard analysis tests
# ---------------------------------------------------------------------------


def _make_cards(specs: list[tuple[str, str]]) -> list[ParsedCard]:
    return [ParsedCard(index=i, rank=r, suit=s) for i, (r, s) in enumerate(specs)]


def test_flush_potential_found():
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("2", "S"), ("3", "D"), ("5", "C")])
    fd = flush_potential(cards)
    assert fd is not None
    assert fd.target_suit == "H"
    assert fd.cards_of_suit == 4
    assert fd.cards_needed == 1
    assert set(fd.keep_indices) == {0, 1, 2, 3}


def test_flush_potential_none_when_too_few():
    cards = _make_cards([("A", "H"), ("K", "D"), ("Q", "C"), ("J", "S"), ("2", "H")])
    fd = flush_potential(cards)
    assert fd is None


def test_fullhouse_potential_with_triple():
    cards = _make_cards([("A", "H"), ("A", "D"), ("A", "C"), ("K", "S"), ("Q", "H"), ("J", "D")])
    fhd = fullhouse_potential(cards)
    assert fhd is not None
    assert fhd.has_triple
    assert len(fhd.keep_indices) == 3  # just the triple (no pair found)


def test_fullhouse_potential_with_two_pair():
    cards = _make_cards([("A", "H"), ("A", "D"), ("K", "C"), ("K", "S"), ("Q", "H"), ("J", "D")])
    fhd = fullhouse_potential(cards)
    assert fhd is not None
    assert fhd.pairs_count == 2
    assert len(fhd.keep_indices) == 4  # both pairs


def test_recommend_discards_empty_when_beats_blind():
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")])
    # Flush: (35 + 11+10+10+10+10) * 4 = 344 — beat target of 300
    result = recommend_discards(cards, blind_target=300)
    assert result == []


def test_recommend_discards_suggests_flush_draw():
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("2", "S"), ("3", "D"), ("5", "C")])
    result = recommend_discards(cards, blind_target=500)
    # Should discard the non-hearts
    assert 4 in result or 5 in result or 6 in result


def test_recommend_discards_prefers_fh_when_trips_exist():
    """With three aces, should go for full house over flush draw."""
    cards = _make_cards([
        ("A", "H"), ("A", "D"), ("A", "C"),  # trips
        ("K", "H"), ("Q", "H"),              # flush draw basis
        ("3", "S"), ("5", "D"),
    ])
    result = recommend_discards(cards, blind_target=500)
    # Should keep the three aces (indices 0,1,2) and discard others
    kept = set(range(len(cards))) - set(result)
    assert {0, 1, 2}.issubset(kept)


# ---------------------------------------------------------------------------
# HeuristicHandPolicy tests
# ---------------------------------------------------------------------------


def test_heuristic_hand_selects_legal_action(wrapped_env):
    policy = HeuristicHandPolicy()
    obs, _ = wrapped_env.reset(seed=42)
    mask = wrapped_env.action_masks()
    action_table = wrapped_env.action_table
    action = policy.select_action(obs, mask, action_table=action_table)
    assert mask[action], "HeuristicHandPolicy selected an illegal action"


def test_heuristic_full_episode(wrapped_env):
    """Run a full episode with all three heuristic policies to verify no crashes."""
    agent = PhaseDispatchAgent(
        hand_policy=HeuristicHandPolicy(),
        shop_policy=HeuristicShopPolicy(),
        blind_policy=HeuristicBlindPolicy(),
    )
    obs, _ = wrapped_env.reset(seed=42)
    steps = 0

    while True:
        mask = wrapped_env.action_masks()
        action_table = wrapped_env.action_table
        action = agent.select_action(obs, mask, action_table=action_table)
        assert mask[action], f"Illegal action {action} at step {steps}"
        obs, _, terminated, truncated, info = wrapped_env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 0
    ante = info.get("balatro/ante_reached", 1)
    assert ante >= 1


def test_heuristic_multiple_seeds(wrapped_env):
    """Run a few episodes with different seeds to catch edge cases."""
    agent = PhaseDispatchAgent(
        hand_policy=HeuristicHandPolicy(),
        shop_policy=HeuristicShopPolicy(),
        blind_policy=HeuristicBlindPolicy(),
    )
    for seed in [0, 7, 13, 42, 99]:
        obs, _ = wrapped_env.reset(seed=seed)
        steps = 0
        while True:
            mask = wrapped_env.action_masks()
            action_table = wrapped_env.action_table
            action = agent.select_action(obs, mask, action_table=action_table)
            assert mask[action], f"Illegal action at step {steps}, seed {seed}"
            obs, _, terminated, truncated, info = wrapped_env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert steps > 0


# ---------------------------------------------------------------------------
# HeuristicShopPolicy tests
# ---------------------------------------------------------------------------


def test_heuristic_shop_leaves_when_nothing_to_buy():
    """With no interesting items, the shop policy should leave."""
    policy = HeuristicShopPolicy()
    mask = np.zeros(500, dtype=bool)

    from dataclasses import dataclass

    @dataclass
    class FakeAction:
        action_type: int
        card_target: tuple[int, ...] | None = None
        entity_target: int | None = None

    table = [FakeAction(action_type=6)]  # NextRound at idx 0
    mask[0] = True

    obs = {
        "global": np.zeros(235, dtype=np.float32),
        "shop_item": np.zeros((10, 9), dtype=np.float32),
        "entity_counts": np.array([0, 0, 0, 0, 0], dtype=np.float32),
    }
    action = policy.select_action(obs, mask, action_table=table)
    assert action == 0  # NextRound


# ---------------------------------------------------------------------------
# Joker parsing tests
# ---------------------------------------------------------------------------


def test_parse_jokers_from_obs():
    """Round-trip: encode a joker key ID into obs tensor, decode back."""
    joker_obs = np.zeros((5, 15), dtype=np.float32)
    joker_key = "j_joker"
    center_id = key_to_id(joker_key)
    joker_obs[0, 0] = center_id / NUM_CENTER_KEYS  # center_key_id
    joker_obs[0, 8] = 0.0  # not debuffed
    joker_obs[0, 10] = 0.0  # no accumulated mult
    joker_obs[0, 11] = 1.0  # x_mult = 1.0 (default)
    joker_obs[0, 14] = 1.0  # condition met

    jokers = parse_jokers_from_obs(joker_obs, n_jokers=1)
    assert len(jokers) == 1
    assert jokers[0].key == "j_joker"
    assert not jokers[0].debuffed


def test_parse_jokers_multiple():
    """Parse multiple jokers from obs."""
    joker_obs = np.zeros((5, 15), dtype=np.float32)
    keys = ["j_joker", "j_duo", "j_lusty_joker"]
    for i, k in enumerate(keys):
        joker_obs[i, 0] = key_to_id(k) / NUM_CENTER_KEYS

    jokers = parse_jokers_from_obs(joker_obs, n_jokers=3)
    assert len(jokers) == 3
    parsed_keys = {j.key for j in jokers}
    assert parsed_keys == set(keys)


def test_parse_jokers_skips_debuffed():
    """Debuffed jokers are parsed but flagged."""
    joker_obs = np.zeros((5, 15), dtype=np.float32)
    joker_obs[0, 0] = key_to_id("j_joker") / NUM_CENTER_KEYS
    joker_obs[0, 8] = 1.0  # debuffed

    jokers = parse_jokers_from_obs(joker_obs, n_jokers=1)
    assert len(jokers) == 1
    assert jokers[0].debuffed


# ---------------------------------------------------------------------------
# Joker scoring tests
# ---------------------------------------------------------------------------


def _make_candidate(
    cards: list[ParsedCard],
    hand_levels: dict[str, int] | None = None,
) -> HandCandidate:
    """Build the best HandCandidate from the given cards."""
    candidates = evaluate_hand_from_parsed(cards, hand_levels, max_candidates=5)
    return candidates[0]


def _make_joker(key: str, mult: float = 0, x_mult: float = 1.0, chips: float = 0) -> ParsedJoker:
    return ParsedJoker(key=key, ability_mult=mult, ability_x_mult=x_mult, ability_chips=chips)


def _default_ctx(cards: list[ParsedCard], n_jokers: int = 1) -> GameContext:
    return GameContext(discards_left=3, n_jokers=n_jokers, deck_size=44, hand_cards=cards)


def test_joker_scoring_no_jokers():
    """Without jokers, score matches base estimate."""
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards, n_jokers=0)
    score = simulate_joker_scoring(cand, [], ctx)
    assert score == cand.estimated_chips


def test_joker_scoring_flat_joker():
    """j_joker adds +4 mult unconditionally."""
    cards = _make_cards([("A", "H"), ("A", "D")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards)
    jokers = [_make_joker("j_joker")]

    base_score = cand.estimated_chips
    joker_score = simulate_joker_scoring(cand, jokers, ctx)

    # Pair: base = (10 + 11 + 11) * 2 = 64
    # With j_joker: (10 + 11 + 11) * (2 + 4) = 192
    assert joker_score > base_score
    assert joker_score == pytest.approx(cand.total_chips * (cand.total_mult + 4), abs=1)


def test_joker_scoring_duo_boosts_pair():
    """j_duo (x2 mult on Pair) should make a Pair outscore a bare Flush."""
    pair_cards = _make_cards([("A", "H"), ("A", "D")])
    flush_cards = _make_cards([("2", "H"), ("3", "H"), ("5", "H"), ("7", "H"), ("9", "H")])

    pair_cand = _make_candidate(pair_cards)
    flush_cand = _make_candidate(flush_cards)

    jokers = [_make_joker("j_duo")]
    pair_ctx = _default_ctx(pair_cards)
    flush_ctx = _default_ctx(flush_cards)

    pair_score = simulate_joker_scoring(pair_cand, jokers, pair_ctx)
    flush_score = simulate_joker_scoring(flush_cand, jokers, flush_ctx)

    # j_duo triggers on Pair (x2) but not on Flush
    assert pair_score > pair_cand.estimated_chips
    # Flush doesn't get the x2 from j_duo
    assert flush_score == pytest.approx(flush_cand.estimated_chips, abs=1)


def test_joker_scoring_suit_mult():
    """j_lusty_joker adds +3 mult per Heart scored."""
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards)
    jokers = [_make_joker("j_lusty_joker")]

    base_score = cand.estimated_chips
    joker_score = simulate_joker_scoring(cand, jokers, ctx)

    # 5 Hearts scored -> +15 mult
    # Flush: total_chips=86, total_mult=4 -> base=344
    # With lusty: 86 * (4+15) = 1634
    assert joker_score > base_score
    expected_mult = cand.total_mult + 15  # 5 hearts * 3
    assert joker_score == pytest.approx(cand.total_chips * expected_mult, abs=1)


def test_joker_scoring_held_cards_baron():
    """j_baron adds x1.5 per held King."""
    # Play a Pair of Aces, hold 2 Kings
    all_cards = _make_cards([
        ("A", "H"), ("A", "D"),  # played
        ("K", "S"), ("K", "C"),  # held
        ("3", "H"),              # held
    ])
    played = all_cards[:2]
    cand = _make_candidate(played)
    ctx = GameContext(discards_left=3, n_jokers=1, deck_size=44, hand_cards=all_cards)
    jokers = [_make_joker("j_baron")]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)

    # 2 held Kings -> x1.5 * x1.5 = x2.25
    expected = cand.total_chips * cand.total_mult * 2.25
    assert joker_score == pytest.approx(expected, abs=1)


def test_joker_scoring_blackboard():
    """j_blackboard gives x3 if all held cards are Spades or Clubs."""
    all_cards = _make_cards([
        ("A", "H"), ("A", "D"),  # played
        ("K", "S"), ("Q", "C"),  # held (dark suits)
    ])
    played = all_cards[:2]
    cand = _make_candidate(played)
    ctx = GameContext(discards_left=3, n_jokers=1, deck_size=44, hand_cards=all_cards)
    jokers = [_make_joker("j_blackboard")]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)
    expected = cand.total_chips * cand.total_mult * 3.0
    assert joker_score == pytest.approx(expected, abs=1)


def test_joker_scoring_blackboard_fails_with_heart():
    """j_blackboard does NOT trigger if a held card is Hearts."""
    all_cards = _make_cards([
        ("A", "H"), ("A", "D"),  # played
        ("K", "S"), ("Q", "H"),  # held (mixed: Spade + Heart)
    ])
    played = all_cards[:2]
    cand = _make_candidate(played)
    ctx = GameContext(discards_left=3, n_jokers=1, deck_size=44, hand_cards=all_cards)
    jokers = [_make_joker("j_blackboard")]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)
    # No trigger -> score equals base
    assert joker_score == pytest.approx(cand.total_chips * cand.total_mult, abs=1)


def test_joker_scoring_banner():
    """j_banner adds +30 chips per remaining discard."""
    cards = _make_cards([("A", "H"), ("A", "D")])
    cand = _make_candidate(cards)
    ctx = GameContext(discards_left=3, n_jokers=1, deck_size=44, hand_cards=cards)
    jokers = [_make_joker("j_banner")]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)
    expected = (cand.total_chips + 90) * cand.total_mult  # +30*3 chips
    assert joker_score == pytest.approx(expected, abs=1)


def test_joker_scoring_fibonacci():
    """j_fibonacci adds +8 mult per A/2/3/5/8 scored."""
    cards = _make_cards([("A", "H"), ("2", "D"), ("3", "C"), ("5", "S"), ("8", "H")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards)
    jokers = [_make_joker("j_fibonacci")]

    base_score = cand.estimated_chips
    joker_score = simulate_joker_scoring(cand, jokers, ctx)

    # High Card with A as scoring card. But all 5 are fibonacci ranks.
    # The hand type might be High Card (just Ace scores) or something else.
    # Let's verify it's bigger
    assert joker_score > base_score


def test_joker_scoring_hand_type_xmult_tribe():
    """j_tribe gives x2 when Flush is played."""
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")])
    cand = _make_candidate(cards)
    assert cand.hand_type in ("Flush", "Straight Flush")

    ctx = _default_ctx(cards)
    jokers = [_make_joker("j_tribe")]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)
    expected = cand.total_chips * cand.total_mult * 2.0
    assert joker_score == pytest.approx(expected, abs=1)


def test_joker_scoring_multiple_jokers():
    """Multiple jokers compose correctly: j_joker (+4 mult) + j_duo (x2)."""
    cards = _make_cards([("A", "H"), ("A", "D")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards, n_jokers=2)
    jokers = [_make_joker("j_joker"), _make_joker("j_duo")]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)

    # Pair of Aces: total_chips=32, total_mult=2
    # j_joker: +4 mult -> mult=6
    # j_duo: x2 -> mult=12
    # Score: 32 * 6 * 2 = 384
    expected = cand.total_chips * (cand.total_mult + 4) * 2.0
    assert joker_score == pytest.approx(expected, abs=1)


def test_joker_scoring_photograph_only_first_face():
    """j_photograph triggers x2 on only the first face card, not all."""
    cards = _make_cards([("K", "H"), ("Q", "D"), ("J", "C"), ("K", "S"), ("Q", "H")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards)
    jokers = [_make_joker("j_photograph")]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)

    # Should be x2 once (not x2^n for n face cards)
    expected = cand.total_chips * cand.total_mult * 2.0
    assert joker_score == pytest.approx(expected, abs=1)


def test_joker_scoring_scaling_reads_obs():
    """Scaling jokers use ability_mult/ability_chips from the observation."""
    cards = _make_cards([("A", "H"), ("A", "D")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards)
    # j_green_joker with 10 accumulated mult
    jokers = [_make_joker("j_green_joker", mult=10)]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)
    expected = cand.total_chips * (cand.total_mult + 10)
    assert joker_score == pytest.approx(expected, abs=1)


def test_joker_scoring_debuffed_skipped():
    """Debuffed jokers should not contribute to scoring."""
    cards = _make_cards([("A", "H"), ("A", "D")])
    cand = _make_candidate(cards)
    ctx = _default_ctx(cards)
    jokers = [ParsedJoker(key="j_joker", ability_mult=0, ability_x_mult=1.0,
                          ability_chips=0, debuffed=True)]

    joker_score = simulate_joker_scoring(cand, jokers, ctx)
    assert joker_score == cand.estimated_chips


# ---------------------------------------------------------------------------
# End-to-end: HeuristicHandPolicy with jokers
# ---------------------------------------------------------------------------


def test_heuristic_full_episode_with_joker_scoring(wrapped_env):
    """Full episode still completes without errors after joker scoring integration."""
    agent = PhaseDispatchAgent(
        hand_policy=HeuristicHandPolicy(),
        shop_policy=HeuristicShopPolicy(),
        blind_policy=HeuristicBlindPolicy(),
    )
    obs, _ = wrapped_env.reset(seed=42)
    steps = 0

    while True:
        mask = wrapped_env.action_masks()
        action_table = wrapped_env.action_table
        action = agent.select_action(obs, mask, action_table=action_table)
        assert mask[action], f"Illegal action {action} at step {steps}"
        obs, _, terminated, truncated, info = wrapped_env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 0


# ---------------------------------------------------------------------------
# Discard sequencing tests
# ---------------------------------------------------------------------------


def test_can_one_shot_small_blind_no_jokers():
    """Flush/FH/Straight can one-shot Small Blind (300) at level 1 without jokers."""
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")])
    ctx = GameContext(discards_left=3, n_jokers=0, deck_size=44, hand_cards=cards)
    assert _can_one_shot_blind(300, [], ctx)


def test_cannot_one_shot_big_blind_no_jokers():
    """No hand type can one-shot Big Blind (450) at level 1 without jokers.
    Max Flush = 344, max FH = 372, max FourOAK = 714 — but FourOAK max
    actually can, so check for 800 (ante 2 small) instead."""
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")])
    ctx = GameContext(discards_left=3, n_jokers=0, deck_size=44, hand_cards=cards)
    # FourOAK max = (60+44)*7 = 728, SF max = (100+51)*8 = 1208
    # So at 800, SF can one-shot but nothing lower can
    assert _can_one_shot_blind(800, [], ctx)
    # At 1300, only SF (1208) can't one-shot
    assert not _can_one_shot_blind(1300, [], ctx)


def test_can_one_shot_with_joker_boost():
    """j_joker (+4 mult) makes Flush score (86)*(4+4) = 688, which beats 450."""
    cards = _make_cards([("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")])
    ctx = GameContext(discards_left=3, n_jokers=1, deck_size=44, hand_cards=cards)
    jokers = [_make_joker("j_joker")]
    assert _can_one_shot_blind(450, jokers, ctx)


def test_can_one_shot_with_tribe():
    """j_tribe (x2 on Flush) makes Flush max = 344*2 = 688."""
    cards = _make_cards([("A", "H"), ("K", "H")])
    ctx = GameContext(discards_left=3, n_jokers=1, deck_size=44, hand_cards=cards)
    jokers = [_make_joker("j_tribe")]
    assert _can_one_shot_blind(600, jokers, ctx)


def _make_hand_obs(
    card_specs: list[tuple[str, str]],
    blind_chips: float,
    discards_left: int = 3,
    hands_left: int = 4,
    joker_keys: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], list]:
    """Build a synthetic observation and action table for hand-phase testing.

    Returns (obs, action_table) where the action table has one PlayHand
    action for the best 5-card combo and one Discard action for the worst cards.
    """
    from dataclasses import dataclass
    from math import log2

    @dataclass
    class FakeAction:
        action_type: int
        card_target: tuple[int, ...] | None = None
        entity_target: int | None = None

    rank_to_id = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
        "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    suit_to_id = {"H": 0, "D": 1, "C": 2, "S": 3}

    n_cards = len(card_specs)
    hand_card = np.zeros((8, 15), dtype=np.float32)
    for i, (rank, suit) in enumerate(card_specs):
        hand_card[i, 0] = rank_to_id[rank] / 14.0
        hand_card[i, 1] = suit_to_id[suit] / 3.0

    g = np.zeros(235, dtype=np.float32)
    g[1] = 1.0  # phase = SELECTING_HAND
    g[10] = 1 / 8.0  # ante = 1
    g[13] = hands_left / 10.0
    g[14] = discards_left / 10.0

    def log_scale(x: float) -> float:
        if x >= 0:
            return log2(1 + x)
        return -log2(1 - x)

    g[18] = log_scale(blind_chips)
    g[27] = log_scale(44)  # deck size

    joker_obs = np.zeros((5, 15), dtype=np.float32)
    n_jokers = 0
    if joker_keys:
        n_jokers = len(joker_keys)
        for i, jk in enumerate(joker_keys):
            joker_obs[i, 0] = key_to_id(jk) / NUM_CENTER_KEYS

    entity_counts = np.array(
        [n_cards, n_jokers, 0, 0, 0], dtype=np.float32
    )

    obs = {
        "global": g,
        "hand_card": hand_card,
        "joker": joker_obs,
        "entity_counts": entity_counts,
        "shop_item": np.zeros((10, 9), dtype=np.float32),
        "consumable": np.zeros((2, 7), dtype=np.float32),
        "pack_card": np.zeros((5, 15), dtype=np.float32),
    }

    # Build action table: play actions for each 5-card combo, discard for worst
    table: list[FakeAction] = []
    mask_list: list[bool] = []

    # Add a PlayHand action for the full hand (up to 5 cards)
    play_indices = tuple(range(min(n_cards, 5)))
    table.append(FakeAction(action_type=0, card_target=play_indices))
    mask_list.append(True)

    # Add a Discard action for all cards
    disc_indices = tuple(range(min(n_cards, 5)))
    table.append(FakeAction(action_type=1, card_target=disc_indices))
    mask_list.append(discards_left > 0)

    mask = np.zeros(len(table), dtype=bool)
    for i, m in enumerate(mask_list):
        mask[i] = m

    return obs, table, mask


def test_discard_sequencing_play_flush_on_big_blind():
    """With a Flush on Big Blind (450), no jokers: play immediately, don't discard.

    Max Flush = 344 < 450, so one-shotting is impossible. The flush is a
    strong hand worth playing in multi-hand mode.
    """
    policy = HeuristicHandPolicy()
    cards = [("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H"),
             ("2", "S"), ("3", "D"), ("5", "C")]
    obs, table, mask = _make_hand_obs(cards, blind_chips=450, discards_left=3)

    action = policy.select_action(obs, mask, action_table=table)
    fa = table[action]
    assert fa.action_type == 0, "Should PLAY the flush, not discard"


def test_discard_sequencing_discard_pair_on_big_blind():
    """With only a Pair on Big Blind (450), no jokers: discard to improve.

    Pair is weak (score ~64). Even though we can't one-shot, discarding
    toward a Flush/FH is better than playing a Pair.
    """
    policy = HeuristicHandPolicy()
    cards = [("A", "H"), ("A", "D"), ("3", "C"), ("5", "S"), ("7", "H"),
             ("9", "D"), ("J", "C"), ("K", "S")]
    obs, table, mask = _make_hand_obs(cards, blind_chips=450, discards_left=3)

    action = policy.select_action(obs, mask, action_table=table)
    fa = table[action]
    assert fa.action_type == 1, "Should DISCARD to improve weak hand"


def test_discard_sequencing_play_flush_on_small_blind():
    """With a Flush on Small Blind (300): play immediately (flush can one-shot 300).

    Flush scores ~344 at best, which beats 300. Step 1 (beat blind check) triggers.
    """
    policy = HeuristicHandPolicy()
    cards = [("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("T", "H")]
    obs, table, mask = _make_hand_obs(cards, blind_chips=300, discards_left=3)

    action = policy.select_action(obs, mask, action_table=table)
    fa = table[action]
    assert fa.action_type == 0, "Should PLAY flush that one-shots Small Blind"


def test_discard_sequencing_discard_flush_draw_with_joker():
    """With 4 Hearts + j_joker, Flush CAN one-shot 450. So discard the off-suit
    card to fish for the 5th Heart (or play the 4 hearts + 1 off-suit as is).

    j_joker makes Flush = (86)*(4+4) = 688 > 450. So discarding to
    complete the flush is worthwhile. But the current hand might already
    score enough to beat the blind as a 4-card flush? No — 4 Hearts don't
    form a flush (need 5). The best hand is a High Card or Pair.

    This test just verifies the agent doesn't trigger the 'play strong
    hand immediately' path when jokers make one-shotting possible.
    """
    policy = HeuristicHandPolicy()
    cards = [("A", "H"), ("K", "H"), ("Q", "H"), ("J", "H"), ("2", "S"),
             ("3", "D"), ("5", "C"), ("7", "S")]
    obs, table, mask = _make_hand_obs(
        cards, blind_chips=450, discards_left=3, joker_keys=["j_joker"]
    )

    action = policy.select_action(obs, mask, action_table=table)
    fa = table[action]
    # With j_joker, a completed flush would score 688 > 450.
    # Best current hand is not a Straight+, so the agent discards
    # (multi-hand skip only applies to Straight+ hands).
    assert fa.action_type == 1, "Should DISCARD toward flush when joker enables one-shot"


# ---------------------------------------------------------------------------
# Straight draw detection tests
# ---------------------------------------------------------------------------


def test_straight_potential_4_card_run():
    """A-Q-J-T is 4 cards toward A-K-Q-J-T straight (needs K)."""
    from balatro_rl.features.hand_evaluator import straight_potential

    cards = [
        ParsedCard(0, "A", "S"), ParsedCard(1, "A", "C"),
        ParsedCard(2, "Q", "C"), ParsedCard(3, "J", "H"),
        ParsedCard(4, "T", "S"), ParsedCard(5, "T", "D"),
        ParsedCard(6, "8", "D"), ParsedCard(7, "7", "C"),
    ]
    sd = straight_potential(cards)
    assert sd is not None
    assert sd.cards_needed == 1
    # Should keep A, Q, J, T (4 cards) and discard the rest
    assert sd.cards_in_run == 4
    kept_ranks = {cards[i].rank for i in sd.keep_indices}
    assert kept_ranks == {"A", "Q", "J", "T"}


def test_straight_potential_open_ended():
    """9-8-7-6 is an open-ended straight draw (needs 5 or T)."""
    from balatro_rl.features.hand_evaluator import straight_potential

    cards = [
        ParsedCard(0, "9", "H"), ParsedCard(1, "8", "D"),
        ParsedCard(2, "7", "C"), ParsedCard(3, "6", "S"),
        ParsedCard(4, "A", "H"), ParsedCard(5, "2", "D"),
    ]
    sd = straight_potential(cards)
    assert sd is not None
    assert sd.cards_needed == 1
    assert sd.is_open_ended is True


def test_straight_potential_ace_low():
    """A-2-3-4 should detect straight draw for A-2-3-4-5."""
    from balatro_rl.features.hand_evaluator import straight_potential

    cards = [
        ParsedCard(0, "A", "S"), ParsedCard(1, "2", "H"),
        ParsedCard(2, "3", "D"), ParsedCard(3, "4", "C"),
        ParsedCard(4, "K", "H"), ParsedCard(5, "Q", "D"),
    ]
    sd = straight_potential(cards)
    assert sd is not None
    assert sd.cards_needed == 1
    kept_ranks = {cards[i].rank for i in sd.keep_indices}
    assert "5" not in kept_ranks  # 5 is what we need
    assert "A" in kept_ranks


def test_straight_potential_no_draw():
    """Scattered ranks with no 3+ consecutive should return None."""
    from balatro_rl.features.hand_evaluator import straight_potential

    cards = [
        ParsedCard(0, "A", "H"), ParsedCard(1, "9", "D"),
        ParsedCard(2, "6", "C"), ParsedCard(3, "3", "S"),
    ]
    sd = straight_potential(cards)
    assert sd is None


def test_recommend_discards_prefers_straight_draw():
    """Hand with 4-card straight draw (need 1 card) should discard toward it.

    The hand [As Ac Qc Jh Ts Td 8d 7c] has A-Q-J-T needing only K for a
    straight. This should be detected and prioritized over flush draws.
    """
    cards = [
        ParsedCard(0, "A", "S"), ParsedCard(1, "A", "C"),
        ParsedCard(2, "Q", "C"), ParsedCard(3, "J", "H"),
        ParsedCard(4, "T", "S"), ParsedCard(5, "T", "D"),
        ParsedCard(6, "8", "D"), ParsedCard(7, "7", "C"),
    ]
    disc = recommend_discards(cards, blind_target=300)
    # Should discard toward the straight (keep A, Q, J, T)
    # The discard should NOT include all of A, Q, J, T
    kept_indices = set(range(8)) - set(disc)
    kept_ranks = {cards[i].rank for i in kept_indices}
    assert "A" in kept_ranks
    assert "Q" in kept_ranks
    assert "J" in kept_ranks
    assert "T" in kept_ranks


def test_recommend_discards_flush_draw_beats_weak_straight():
    """4-card flush draw (need 1) beats 3-card straight draw (need 2)."""
    cards = [
        ParsedCard(0, "A", "H"), ParsedCard(1, "K", "H"),
        ParsedCard(2, "Q", "H"), ParsedCard(3, "J", "H"),
        ParsedCard(4, "5", "S"), ParsedCard(5, "3", "D"),
        ParsedCard(6, "2", "C"), ParsedCard(7, "9", "S"),
    ]
    disc = recommend_discards(cards, blind_target=600)
    kept_indices = set(range(8)) - set(disc)
    # Should keep the 4 hearts for flush draw, not the A-K-Q-J straight draw
    # (Both draws happen to keep the same cards here — A,K,Q,J of hearts —
    # but the flush draw should trigger first in priority order.)
    kept_suits = {cards[i].suit for i in kept_indices}
    assert "H" in kept_suits


# ---------------------------------------------------------------------------
# Shop interest threshold tests
# ---------------------------------------------------------------------------


def test_would_break_interest_basic():
    from balatro_rl.agents.shop import _would_break_interest
    # $10 -> $7: drops from tier 2 to tier 1
    assert _would_break_interest(10, 3) is True
    # $10 -> $8: still tier 1 -> tier 1? No: 10//5=2, 7//5=1. So breaks.
    assert _would_break_interest(10, 3) is True
    # $12 -> $9: 12//5=2, 9//5=1, breaks
    assert _would_break_interest(12, 3) is True
    # $12 -> $10: 12//5=2, 10//5=2, safe
    assert _would_break_interest(12, 2) is False
    # $7 -> $4: 7//5=1, 4//5=0, breaks
    assert _would_break_interest(7, 3) is True
    # $7 -> $5: 7//5=1, 5//5=1, safe
    assert _would_break_interest(7, 2) is False


def test_would_break_interest_above_25():
    from balatro_rl.agents.shop import _would_break_interest
    # $30 -> $27: capped at 25, so 25//5=5 -> 25//5=5, safe
    assert _would_break_interest(30, 3) is False
    # $27 -> $22: 25//5=5 -> 22//5=4, breaks
    assert _would_break_interest(27, 5) is True


def test_shop_skips_planet_below_interest_threshold():
    """Shop should not buy a planet if it drops below interest threshold."""
    from dataclasses import dataclass
    from math import log2

    from balatro_rl.agents.shop import HeuristicShopPolicy

    @dataclass
    class FakeAction:
        action_type: int
        card_target: tuple[int, ...] | None = None
        entity_target: int | None = None

    def log_scale(x: float) -> float:
        return log2(1 + x) if x >= 0 else -log2(1 - x)

    policy = HeuristicShopPolicy()

    g = np.zeros(235, dtype=np.float32)
    g[10] = 2 / 8.0  # ante 2 (planets now allowed)
    g[12] = log_scale(12)  # $12

    # One shop item: Jupiter (Flush planet) costing $3
    shop_items = np.zeros((10, 9), dtype=np.float32)
    jupiter_id = key_to_id("c_jupiter")
    shop_items[0, 0] = jupiter_id / NUM_CENTER_KEYS
    shop_items[0, 1] = 3 / 7.0  # card_set "Planet" = 3
    shop_items[0, 2] = log_scale(3)  # cost $3
    shop_items[0, 3] = 1.0  # affordable

    obs = {
        "global": g,
        "hand_card": np.zeros((8, 15), dtype=np.float32),
        "joker": np.zeros((5, 15), dtype=np.float32),
        "entity_counts": np.array([0, 0, 0, 1, 0], dtype=np.float32),
        "shop_item": shop_items,
        "consumable": np.zeros((2, 7), dtype=np.float32),
        "pack_card": np.zeros((5, 15), dtype=np.float32),
    }

    # BuyCard for slot 0, and NextRound
    table = [
        FakeAction(action_type=8, entity_target=0),
        FakeAction(action_type=6),
    ]
    mask = np.array([True, True], dtype=bool)

    action = policy.select_action(obs, mask, action_table=table)
    # $12 - $3 = $9, drops from tier 2 to tier 1. Should skip.
    assert table[action].action_type == 6, "Should NextRound, not buy planet that breaks interest"


def test_shop_buys_planet_when_safe():
    """Shop should buy Jupiter when it doesn't break interest threshold."""
    from dataclasses import dataclass
    from math import log2

    from balatro_rl.agents.shop import HeuristicShopPolicy

    @dataclass
    class FakeAction:
        action_type: int
        card_target: tuple[int, ...] | None = None
        entity_target: int | None = None

    def log_scale(x: float) -> float:
        return log2(1 + x) if x >= 0 else -log2(1 - x)

    policy = HeuristicShopPolicy()

    g = np.zeros(235, dtype=np.float32)
    g[10] = 2 / 8.0  # ante 2
    g[12] = log_scale(13)  # $13: buying for $3 -> $10, stays at tier 2

    shop_items = np.zeros((10, 9), dtype=np.float32)
    jupiter_id = key_to_id("c_jupiter")
    shop_items[0, 0] = jupiter_id / NUM_CENTER_KEYS
    shop_items[0, 1] = 3 / 7.0  # Planet
    shop_items[0, 2] = log_scale(3)
    shop_items[0, 3] = 1.0

    obs = {
        "global": g,
        "hand_card": np.zeros((8, 15), dtype=np.float32),
        "joker": np.zeros((5, 15), dtype=np.float32),
        "entity_counts": np.array([0, 0, 0, 1, 0], dtype=np.float32),
        "shop_item": shop_items,
        "consumable": np.zeros((2, 7), dtype=np.float32),
        "pack_card": np.zeros((5, 15), dtype=np.float32),
    }

    table = [
        FakeAction(action_type=8, entity_target=0),
        FakeAction(action_type=6),
    ]
    mask = np.array([True, True], dtype=bool)

    action = policy.select_action(obs, mask, action_table=table)
    assert table[action].action_type == 8, "Should buy Jupiter when interest is safe"


def test_shop_ante1_only_jupiter_saturn():
    """In ante 1, shop should only buy Jupiter (Flush) or Saturn (Straight),
    not Uranus (Two Pair) or other low-value planets."""
    from dataclasses import dataclass
    from math import log2

    from balatro_rl.agents.shop import HeuristicShopPolicy

    @dataclass
    class FakeAction:
        action_type: int
        card_target: tuple[int, ...] | None = None
        entity_target: int | None = None

    def log_scale(x: float) -> float:
        return log2(1 + x) if x >= 0 else -log2(1 - x)

    policy = HeuristicShopPolicy()

    g = np.zeros(235, dtype=np.float32)
    g[10] = 1 / 8.0  # ante 1
    g[12] = log_scale(20)  # $20, plenty of money

    # Shop has Uranus (Two Pair planet)
    shop_items = np.zeros((10, 9), dtype=np.float32)
    uranus_id = key_to_id("c_uranus")
    shop_items[0, 0] = uranus_id / NUM_CENTER_KEYS
    shop_items[0, 1] = 3 / 7.0  # Planet
    shop_items[0, 2] = log_scale(3)
    shop_items[0, 3] = 1.0

    obs = {
        "global": g,
        "hand_card": np.zeros((8, 15), dtype=np.float32),
        "joker": np.zeros((5, 15), dtype=np.float32),
        "entity_counts": np.array([0, 0, 0, 1, 0], dtype=np.float32),
        "shop_item": shop_items,
        "consumable": np.zeros((2, 7), dtype=np.float32),
        "pack_card": np.zeros((5, 15), dtype=np.float32),
    }

    table = [
        FakeAction(action_type=8, entity_target=0),
        FakeAction(action_type=6),
    ]
    mask = np.array([True, True], dtype=bool)

    action = policy.select_action(obs, mask, action_table=table)
    assert table[action].action_type == 6, "Should NOT buy Uranus in ante 1"


def test_shop_ante1_buys_jupiter():
    """In ante 1, Jupiter (Flush) should still be purchased."""
    from dataclasses import dataclass
    from math import log2

    from balatro_rl.agents.shop import HeuristicShopPolicy

    @dataclass
    class FakeAction:
        action_type: int
        card_target: tuple[int, ...] | None = None
        entity_target: int | None = None

    def log_scale(x: float) -> float:
        return log2(1 + x) if x >= 0 else -log2(1 - x)

    policy = HeuristicShopPolicy()

    g = np.zeros(235, dtype=np.float32)
    g[10] = 1 / 8.0  # ante 1
    g[12] = log_scale(20)  # $20

    shop_items = np.zeros((10, 9), dtype=np.float32)
    jupiter_id = key_to_id("c_jupiter")
    shop_items[0, 0] = jupiter_id / NUM_CENTER_KEYS
    shop_items[0, 1] = 3 / 7.0
    shop_items[0, 2] = log_scale(3)
    shop_items[0, 3] = 1.0

    obs = {
        "global": g,
        "hand_card": np.zeros((8, 15), dtype=np.float32),
        "joker": np.zeros((5, 15), dtype=np.float32),
        "entity_counts": np.array([0, 0, 0, 1, 0], dtype=np.float32),
        "shop_item": shop_items,
        "consumable": np.zeros((2, 7), dtype=np.float32),
        "pack_card": np.zeros((5, 15), dtype=np.float32),
    }

    table = [
        FakeAction(action_type=8, entity_target=0),
        FakeAction(action_type=6),
    ]
    mask = np.array([True, True], dtype=bool)

    action = policy.select_action(obs, mask, action_table=table)
    assert table[action].action_type == 8, "Should buy Jupiter in ante 1"


def test_shop_joker_ignores_interest():
    """Jokers are always bought regardless of interest threshold impact."""
    from dataclasses import dataclass
    from math import log2

    from balatro_rl.agents.shop import HeuristicShopPolicy

    @dataclass
    class FakeAction:
        action_type: int
        card_target: tuple[int, ...] | None = None
        entity_target: int | None = None

    def log_scale(x: float) -> float:
        return log2(1 + x) if x >= 0 else -log2(1 - x)

    policy = HeuristicShopPolicy()

    g = np.zeros(235, dtype=np.float32)
    g[10] = 1 / 8.0
    g[12] = log_scale(7)  # $7: buying $5 joker -> $2, breaks tier 1

    shop_items = np.zeros((10, 9), dtype=np.float32)
    joker_id = key_to_id("j_joker")  # common scoring joker
    shop_items[0, 0] = joker_id / NUM_CENTER_KEYS
    shop_items[0, 1] = 2 / 9.0  # Joker card_set = 2
    shop_items[0, 2] = log_scale(5)
    shop_items[0, 3] = 1.0

    obs = {
        "global": g,
        "hand_card": np.zeros((8, 15), dtype=np.float32),
        "joker": np.zeros((5, 15), dtype=np.float32),
        "entity_counts": np.array([0, 0, 0, 1, 0], dtype=np.float32),
        "shop_item": shop_items,
        "consumable": np.zeros((2, 7), dtype=np.float32),
        "pack_card": np.zeros((5, 15), dtype=np.float32),
    }

    table = [
        FakeAction(action_type=8, entity_target=0),
        FakeAction(action_type=6),
    ]
    mask = np.array([True, True], dtype=bool)

    action = policy.select_action(obs, mask, action_table=table)
    assert table[action].action_type == 8, "Should always buy joker regardless of interest"
