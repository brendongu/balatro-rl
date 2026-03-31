"""Tests for the expert gameplay capture harness."""

import json

import numpy as np
import pytest

from balatro_rl.capture.observer import _card_ids, _states_differ, infer_action
from balatro_rl.capture.recorder import SessionRecorder
from balatro_rl.capture.scenarios import CardSpec, Scenario, load_scenario
from balatro_rl.capture.state_builder import (
    _build_card,
    _build_hand_levels,
    _build_playing_card,
    build_game_state,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal balatrobot JSON structures
# ---------------------------------------------------------------------------


def _make_bot_card(
    card_id: int,
    *,
    suit: str = "H",
    rank: str = "A",
    card_set: str = "DEFAULT",
    key: str | None = None,
    label: str | None = None,
    edition: str | None = None,
    seal: str | None = None,
    enhancement: str | None = None,
    debuff: bool = False,
    buy_cost: int = 0,
    sell_cost: int = 0,
) -> dict:
    """Build a minimal balatrobot card JSON."""
    if key is None:
        key = f"{suit}_{rank}" if card_set == "DEFAULT" else f"j_joker"
    return {
        "id": card_id,
        "key": key,
        "set": card_set,
        "label": label or key,
        "value": {"suit": suit, "rank": rank, "effect": ""},
        "modifier": {
            "seal": seal,
            "edition": edition,
            "enhancement": enhancement,
            "eternal": False,
            "perishable": None,
            "rental": False,
        },
        "state": {"debuff": debuff, "hidden": False, "highlight": False},
        "cost": {"sell": sell_cost, "buy": buy_cost},
    }


def _make_area(cards: list[dict], limit: int = 8) -> dict:
    return {"count": len(cards), "limit": limit, "highlighted_limit": 5, "cards": cards}


def _make_gamestate(
    *,
    state: str = "SELECTING_HAND",
    ante: int = 1,
    money: int = 4,
    hand_cards: list[dict] | None = None,
    joker_cards: list[dict] | None = None,
    shop_cards: list[dict] | None = None,
    pack_cards: list[dict] | None = None,
    consumable_cards: list[dict] | None = None,
    hands_left: int = 4,
    discards_left: int = 3,
    chips: int = 0,
) -> dict:
    """Build a minimal balatrobot gamestate JSON."""
    return {
        "state": state,
        "ante_num": ante,
        "round_num": 1,
        "money": money,
        "deck": "RED",
        "stake": "WHITE",
        "seed": "TEST",
        "won": False,
        "used_vouchers": {},
        "hands": {},
        "round": {
            "hands_left": hands_left,
            "hands_played": 0,
            "discards_left": discards_left,
            "discards_used": 0,
            "reroll_cost": 5,
            "chips": chips,
        },
        "blinds": {
            "small": {"type": "SMALL", "status": "CURRENT", "name": "Small Blind", "effect": "", "score": 300, "tag_name": "", "tag_effect": ""},
            "big": {"type": "BIG", "status": "UPCOMING", "name": "Big Blind", "effect": "", "score": 450, "tag_name": "", "tag_effect": ""},
            "boss": {"type": "BOSS", "status": "UPCOMING", "name": "The Hook", "effect": "", "score": 600, "tag_name": "", "tag_effect": ""},
        },
        "hand": _make_area(hand_cards or [], limit=8),
        "jokers": _make_area(joker_cards or [], limit=5),
        "consumables": _make_area(consumable_cards or [], limit=2),
        "cards": _make_area([], limit=44),
        "shop": _make_area(shop_cards or [], limit=2),
        "vouchers": _make_area([], limit=1),
        "packs": _make_area([], limit=2),
        "pack": _make_area(pack_cards or [], limit=0),
    }


# ---------------------------------------------------------------------------
# state_builder tests
# ---------------------------------------------------------------------------


class TestStateBuilder:
    def test_build_playing_card_basic(self):
        card_json = _make_bot_card(1, suit="S", rank="K")
        card = _build_playing_card(card_json)

        assert card.base is not None
        assert card.base.suit.value == "Spades"
        assert card.base.rank.value == "King"
        assert card.sort_id == 1

    def test_build_playing_card_with_modifiers(self):
        card_json = _make_bot_card(
            2, suit="H", rank="A",
            edition="FOIL", seal="GOLD", enhancement="GLASS",
        )
        card = _build_playing_card(card_json)

        assert card.base.suit.value == "Hearts"
        assert card.base.rank.value == "Ace"
        assert card.edition is not None
        assert card.edition.get("foil") is True
        assert card.seal == "Gold"

    def test_build_playing_card_debuff(self):
        card_json = _make_bot_card(3, debuff=True)
        card = _build_playing_card(card_json)
        assert card.debuff is True

    def test_build_joker(self):
        card_json = _make_bot_card(
            10, card_set="JOKER", key="j_joker",
            label="Joker",
        )
        card = _build_card(card_json)

        assert card.ability.get("set") == "Joker"
        assert card.center_key == "j_joker"
        assert card.sort_id == 10

    def test_build_consumable(self):
        card_json = _make_bot_card(
            20, card_set="TAROT", key="c_fool",
            label="The Fool",
        )
        card = _build_card(card_json)

        assert card.ability.get("set") == "Tarot"
        assert card.center_key == "c_fool"

    def test_build_hand_levels(self):
        hands_json = {
            "Flush": {"order": 1, "level": 3, "chips": 55, "mult": 8, "played": 5, "played_this_round": 1},
            "Pair": {"order": 8, "level": 2, "chips": 15, "mult": 3, "played": 10, "played_this_round": 0},
        }
        from jackdaw.engine.data.hands import HandType

        hl = _build_hand_levels(hands_json)
        flush_state = hl.get_state(HandType.FLUSH)
        assert flush_state.level == 3
        assert flush_state.chips == 55
        assert flush_state.mult == 8
        assert flush_state.played == 5

    def test_build_game_state_full(self):
        hand = [
            _make_bot_card(1, suit="H", rank="A"),
            _make_bot_card(2, suit="S", rank="K"),
            _make_bot_card(3, suit="D", rank="Q"),
        ]
        jokers = [
            _make_bot_card(10, card_set="JOKER", key="j_joker"),
        ]
        gs_json = _make_gamestate(hand_cards=hand, joker_cards=jokers)
        gs = build_game_state(gs_json)

        # Phase
        from jackdaw.engine.actions import GamePhase
        assert gs["phase"] == GamePhase.SELECTING_HAND

        # Cards are real Card objects
        assert len(gs["hand"]) == 3
        assert gs["hand"][0].base is not None
        assert gs["hand"][0].base.rank.value == "Ace"

        assert len(gs["jokers"]) == 1
        assert gs["jokers"][0].center_key == "j_joker"

        # Scalar fields
        assert gs["dollars"] == 4
        assert gs["current_round"]["hands_left"] == 4

    def test_build_game_state_encodes(self):
        """Verify that build_game_state produces a dict that encode_observation can consume."""
        from jackdaw.env.observation import encode_observation

        hand = [_make_bot_card(i, suit="H", rank=r) for i, r in enumerate(["A", "K", "Q", "J", "T"])]
        gs_json = _make_gamestate(hand_cards=hand)
        gs = build_game_state(gs_json)

        obs = encode_observation(gs)
        assert obs.global_context.shape[0] > 0
        assert obs.hand_cards.shape[0] == 5


# ---------------------------------------------------------------------------
# recorder tests
# ---------------------------------------------------------------------------


class TestRecorder:
    def test_session_lifecycle(self, tmp_path):
        recorder = SessionRecorder(save_dir=tmp_path)

        path = recorder.begin_session(mode="test")
        assert path.suffix == ".jsonl"
        assert recorder.in_session

        gs = _make_gamestate()
        recorder.record_transition(gs, action={"method": "select", "params": {}})
        recorder.record_transition(gs)

        result_path = recorder.end_session(ante_reached=2, won=False)
        assert not recorder.in_session
        assert result_path.exists()
        assert recorder.sessions_saved == 1

        # Verify JSONL contents
        lines = result_path.read_text().strip().split("\n")
        assert len(lines) == 4  # start + 2 transitions + end

        start = json.loads(lines[0])
        assert start["type"] == "session_start"
        assert start["mode"] == "test"

        t1 = json.loads(lines[1])
        assert t1["type"] == "transition"
        assert t1["action"]["method"] == "select"

        end = json.loads(lines[-1])
        assert end["type"] == "session_end"
        assert end["ante_reached"] == 2

    def test_multiple_sessions(self, tmp_path):
        recorder = SessionRecorder(save_dir=tmp_path)

        recorder.begin_session()
        recorder.end_session()

        recorder.begin_session()
        recorder.end_session()

        assert recorder.sessions_saved == 2
        assert len(list(tmp_path.glob("*.jsonl"))) == 2


# ---------------------------------------------------------------------------
# observer / action inference tests
# ---------------------------------------------------------------------------


class TestActionInference:
    def test_select_blind(self):
        prev = _make_gamestate(state="BLIND_SELECT")
        curr = _make_gamestate(state="SELECTING_HAND")
        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "select"

    def test_skip_blind(self):
        prev = _make_gamestate(state="BLIND_SELECT", ante=1)
        curr = _make_gamestate(state="BLIND_SELECT", ante=1)
        # Skip advances to next blind but stays in BLIND_SELECT
        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "skip"

    def test_cash_out(self):
        prev = _make_gamestate(state="ROUND_EVAL")
        curr = _make_gamestate(state="SHOP")
        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "cash_out"

    def test_next_round(self):
        prev = _make_gamestate(state="SHOP")
        curr = _make_gamestate(state="BLIND_SELECT")
        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "next_round"

    def test_play_hand(self):
        hand_before = [
            _make_bot_card(1, suit="H", rank="A"),
            _make_bot_card(2, suit="S", rank="K"),
            _make_bot_card(3, suit="D", rank="Q"),
            _make_bot_card(4, suit="C", rank="J"),
            _make_bot_card(5, suit="H", rank="T"),
        ]
        hand_after = [
            _make_bot_card(3, suit="D", rank="Q"),
            _make_bot_card(4, suit="C", rank="J"),
        ]
        prev = _make_gamestate(hand_cards=hand_before, chips=0, hands_left=4)
        curr = _make_gamestate(hand_cards=hand_after, chips=150, hands_left=3)

        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "play"
        assert sorted(action["params"]["cards"]) == [0, 1, 4]

    def test_discard(self):
        hand_before = [
            _make_bot_card(1, suit="H", rank="A"),
            _make_bot_card(2, suit="S", rank="K"),
            _make_bot_card(3, suit="D", rank="Q"),
        ]
        hand_after = [
            _make_bot_card(1, suit="H", rank="A"),
        ]
        prev = _make_gamestate(hand_cards=hand_before, chips=0, hands_left=4, discards_left=3)
        curr = _make_gamestate(hand_cards=hand_after, chips=0, hands_left=4, discards_left=2)

        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "discard"
        assert sorted(action["params"]["cards"]) == [1, 2]

    def test_buy_shop_card(self):
        shop_before = [
            _make_bot_card(100, card_set="JOKER", key="j_joker", buy_cost=4),
            _make_bot_card(101, card_set="JOKER", key="j_banner", buy_cost=5),
        ]
        shop_after = [
            _make_bot_card(101, card_set="JOKER", key="j_banner", buy_cost=5),
        ]
        prev = _make_gamestate(state="SHOP", money=10, shop_cards=shop_before)
        curr = _make_gamestate(state="SHOP", money=6, shop_cards=shop_after)

        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "buy"
        assert action["params"]["card"] == 0

    def test_reroll(self):
        shop_before = [_make_bot_card(100, card_set="JOKER", key="j_joker")]
        shop_after = [_make_bot_card(200, card_set="JOKER", key="j_banner")]
        prev = _make_gamestate(state="SHOP", money=10, shop_cards=shop_before)
        curr = _make_gamestate(state="SHOP", money=5, shop_cards=shop_after)

        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "reroll"

    def test_no_change(self):
        gs = _make_gamestate()
        action = infer_action(gs, gs)
        assert action is None

    def test_sell_joker(self):
        jokers_before = [
            _make_bot_card(10, card_set="JOKER", key="j_joker"),
            _make_bot_card(11, card_set="JOKER", key="j_banner"),
        ]
        jokers_after = [
            _make_bot_card(11, card_set="JOKER", key="j_banner"),
        ]
        prev = _make_gamestate(state="SHOP", money=5, joker_cards=jokers_before)
        curr = _make_gamestate(state="SHOP", money=7, joker_cards=jokers_after)

        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "sell"
        assert action["params"]["joker"] == 0

    def test_pack_pick(self):
        pack_before = [
            _make_bot_card(50, card_set="JOKER", key="j_joker"),
            _make_bot_card(51, card_set="JOKER", key="j_banner"),
        ]
        pack_after = [
            _make_bot_card(51, card_set="JOKER", key="j_banner"),
        ]
        prev = _make_gamestate(state="SMODS_BOOSTER_OPENED", pack_cards=pack_before)
        curr = _make_gamestate(state="SMODS_BOOSTER_OPENED", pack_cards=pack_after)

        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "pack"
        assert action["params"]["card"] == 0

    def test_pack_skip(self):
        pack_cards = [_make_bot_card(50, card_set="JOKER", key="j_joker")]
        prev = _make_gamestate(state="SMODS_BOOSTER_OPENED", pack_cards=pack_cards)
        curr = _make_gamestate(state="SHOP")

        action = infer_action(prev, curr)
        assert action is not None
        assert action["method"] == "pack"
        assert action["params"].get("skip") is True


class TestStatesDiffer:
    def test_identical_states(self):
        gs = _make_gamestate()
        assert not _states_differ(gs, gs)

    def test_phase_change(self):
        a = _make_gamestate(state="BLIND_SELECT")
        b = _make_gamestate(state="SELECTING_HAND")
        assert _states_differ(a, b)

    def test_money_change(self):
        a = _make_gamestate(money=10)
        b = _make_gamestate(money=5)
        assert _states_differ(a, b)

    def test_hand_change(self):
        a = _make_gamestate(hand_cards=[_make_bot_card(1)])
        b = _make_gamestate(hand_cards=[_make_bot_card(2)])
        assert _states_differ(a, b)


# ---------------------------------------------------------------------------
# scenario tests
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_load_scenario(self, tmp_path):
        toml_content = """
[meta]
name = "test_scenario"
description = "A test scenario"

[game]
deck = "BLUE"
stake = "RED"
seed = "TEST_123"

[state]
ante = 3
money = 20

[[jokers]]
key = "j_joker"

[[jokers]]
key = "j_blueprint"
edition = "FOIL"

[[consumables]]
key = "c_fool"

[[cards]]
key = "H_A"
seal = "GOLD"
"""
        scenario_path = tmp_path / "test.toml"
        scenario_path.write_text(toml_content)

        scenario = load_scenario(scenario_path)

        assert scenario.name == "test_scenario"
        assert scenario.description == "A test scenario"
        assert scenario.deck == "BLUE"
        assert scenario.stake == "RED"
        assert scenario.seed == "TEST_123"
        assert scenario.ante == 3
        assert scenario.money == 20
        assert len(scenario.jokers) == 2
        assert scenario.jokers[0].key == "j_joker"
        assert scenario.jokers[1].key == "j_blueprint"
        assert scenario.jokers[1].edition == "FOIL"
        assert len(scenario.consumables) == 1
        assert scenario.consumables[0].key == "c_fool"
        assert len(scenario.cards) == 1
        assert scenario.cards[0].key == "H_A"
        assert scenario.cards[0].seal == "GOLD"

    def test_load_minimal_scenario(self, tmp_path):
        toml_content = """
[game]
deck = "RED"
"""
        scenario_path = tmp_path / "minimal.toml"
        scenario_path.write_text(toml_content)

        scenario = load_scenario(scenario_path)

        assert scenario.deck == "RED"
        assert scenario.stake == "WHITE"
        assert scenario.seed is None
        assert scenario.ante is None
        assert scenario.money is None
        assert len(scenario.jokers) == 0

    def test_card_spec_dataclass(self):
        spec = CardSpec(key="j_joker", edition="FOIL", eternal=True)
        assert spec.key == "j_joker"
        assert spec.edition == "FOIL"
        assert spec.eternal is True
        assert spec.seal is None
