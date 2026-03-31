"""Convert balatrobot JSON gamestates into engine game_state dicts.

The existing ``bot_state_to_game_state()`` in jackdaw only extracts keys
(no Card objects), so ``encode_observation()`` cannot consume its output.
This module constructs proper engine game_state dicts with real Card
objects from balatrobot's rich JSON, enabling observation encoding
identical to DirectAdapter training runs.

Usage::

    from balatro_rl.capture.state_builder import build_game_state
    from jackdaw.env.observation import encode_observation

    gs = build_game_state(bot_json)
    obs = encode_observation(gs)
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import (
    RANK_LETTER,
    SUIT_LETTER,
    create_consumable,
    create_joker,
    create_playing_card,
    create_voucher,
)
from jackdaw.engine.hand_levels import HandLevels

_STATE_MAP: dict[str, GamePhase] = {
    "BLIND_SELECT": GamePhase.BLIND_SELECT,
    "SELECTING_HAND": GamePhase.SELECTING_HAND,
    "ROUND_EVAL": GamePhase.ROUND_EVAL,
    "SHOP": GamePhase.SHOP,
    "SMODS_BOOSTER_OPENED": GamePhase.PACK_OPENING,
    "GAME_OVER": GamePhase.GAME_OVER,
}

_EDITION_FROM_BOT: dict[str, str] = {
    "FOIL": "foil",
    "HOLO": "holo",
    "POLYCHROME": "polychrome",
    "NEGATIVE": "negative",
}

_ENHANCEMENT_FROM_BOT: dict[str, str] = {
    "BONUS": "m_bonus",
    "MULT": "m_mult",
    "WILD": "m_wild",
    "GLASS": "m_glass",
    "STEEL": "m_steel",
    "STONE": "m_stone",
    "GOLD": "m_gold",
    "LUCKY": "m_lucky",
}

_SEAL_FROM_BOT: dict[str, str] = {
    "GOLD": "Gold",
    "RED": "Red",
    "BLUE": "Blue",
    "PURPLE": "Purple",
}

_RANK_FROM_BOT: dict[str, str] = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "T",
    "J": "J", "Q": "Q", "K": "K", "A": "A",
}

_SUIT_FROM_BOT: dict[str, str] = {
    "H": "H", "D": "D", "C": "C", "S": "S",
}


def _build_playing_card(card_json: dict[str, Any]) -> Card:
    """Construct a playing Card from a balatrobot card JSON object."""
    value = card_json.get("value", {})
    suit_letter = value.get("suit", "H")
    rank_letter = value.get("rank", "A")

    suit = SUIT_LETTER[_SUIT_FROM_BOT.get(suit_letter, suit_letter)]
    rank = RANK_LETTER[_RANK_FROM_BOT.get(rank_letter, rank_letter)]

    modifier = card_json.get("modifier", {})
    enhancement_bot = modifier.get("enhancement")
    enhancement = _ENHANCEMENT_FROM_BOT.get(enhancement_bot, "c_base") if enhancement_bot else "c_base"

    edition_bot = modifier.get("edition")
    edition = {_EDITION_FROM_BOT[edition_bot]: True} if edition_bot and edition_bot in _EDITION_FROM_BOT else None

    seal_bot = modifier.get("seal")
    seal = _SEAL_FROM_BOT.get(seal_bot) if seal_bot else None

    card = create_playing_card(suit, rank, enhancement=enhancement, edition=edition, seal=seal)

    state = card_json.get("state", {})
    card.debuff = state.get("debuff", False)
    if state.get("hidden", False):
        card.facing = "back"

    card.sort_id = card_json.get("id", card.sort_id)

    cost_info = card_json.get("cost", {})
    card.cost = cost_info.get("buy", card.cost)
    card.sell_cost = cost_info.get("sell", card.sell_cost)

    return card


def _build_joker(card_json: dict[str, Any]) -> Card:
    """Construct a joker Card from a balatrobot card JSON object."""
    key = card_json.get("key", "j_joker")

    modifier = card_json.get("modifier", {})
    edition_bot = modifier.get("edition")
    edition = {_EDITION_FROM_BOT[edition_bot]: True} if edition_bot and edition_bot in _EDITION_FROM_BOT else None

    eternal = modifier.get("eternal", False) or False
    perishable = modifier.get("perishable")
    rental = modifier.get("rental", False) or False

    card = create_joker(key, edition=edition, eternal=eternal, perishable=bool(perishable), rental=rental)

    if perishable is not None and isinstance(perishable, int):
        card.perish_tally = perishable

    card.sort_id = card_json.get("id", card.sort_id)
    card.debuff = card_json.get("state", {}).get("debuff", False)

    cost_info = card_json.get("cost", {})
    card.cost = cost_info.get("buy", card.cost)
    card.sell_cost = cost_info.get("sell", card.sell_cost)

    return card


def _build_consumable(card_json: dict[str, Any]) -> Card:
    """Construct a consumable Card from a balatrobot card JSON object."""
    key = card_json.get("key", "c_fool")
    card = create_consumable(key)
    card.sort_id = card_json.get("id", card.sort_id)

    cost_info = card_json.get("cost", {})
    card.cost = cost_info.get("buy", card.cost)
    card.sell_cost = cost_info.get("sell", card.sell_cost)
    return card


def _build_card(card_json: dict[str, Any]) -> Card:
    """Build a Card from balatrobot JSON, dispatching by card set."""
    card_set = card_json.get("set", "DEFAULT")
    key = card_json.get("key", "")

    if card_set == "JOKER" or key.startswith("j_"):
        return _build_joker(card_json)
    elif card_set in ("TAROT", "PLANET", "SPECTRAL") or key.startswith("c_"):
        return _build_consumable(card_json)
    elif card_set == "VOUCHER" or key.startswith("v_"):
        card = create_voucher(key)
        card.sort_id = card_json.get("id", card.sort_id)
        return card
    elif card_set == "BOOSTER" or key.startswith("p_"):
        card = Card()
        card.set_ability(key)
        card.sort_id = card_json.get("id", card.sort_id)
        cost_info = card_json.get("cost", {})
        card.cost = cost_info.get("buy", card.cost)
        card.sell_cost = cost_info.get("sell", card.sell_cost)
        return card
    else:
        return _build_playing_card(card_json)


def _build_cards_from_area(area_json: dict[str, Any]) -> list[Card]:
    """Build a list of Cards from a balatrobot area JSON (hand, jokers, etc.)."""
    return [_build_card(c) for c in area_json.get("cards", [])]


def _build_hand_levels(hands_json: dict[str, Any]) -> HandLevels:
    """Build HandLevels from balatrobot's hands JSON."""
    from jackdaw.engine.data.hands import HandType

    hl = HandLevels()
    for hand_name, hand_data in hands_json.items():
        try:
            ht = HandType(hand_name)
        except ValueError:
            continue
        state = hl.get_state(ht)
        state.level = hand_data.get("level", 1)
        state.chips = hand_data.get("chips", state.chips)
        state.mult = hand_data.get("mult", state.mult)
        state.played = hand_data.get("played", 0)
        state.played_this_round = hand_data.get("played_this_round", 0)
    return hl


def _build_blind(blinds_json: dict[str, Any], gs: dict[str, Any]) -> Any:
    """Build a Blind object for the current active blind."""
    from jackdaw.engine.blind import Blind
    from jackdaw.engine.data.prototypes import BLINDS

    for btype in ("small", "big", "boss"):
        bi = blinds_json.get(btype, {})
        if bi.get("status") == "CURRENT":
            name = bi.get("name", "")
            score = bi.get("score", 0)
            for bk, proto in BLINDS.items():
                if proto.name == name:
                    ante = gs.get("round_resets", {}).get("ante", 1)
                    blind = Blind.create(bk, ante, 1, 1.0)
                    blind.chips = score
                    return blind
            blind = Blind.create("bl_small", 1, 1, 1.0)
            blind.chips = score
            return blind
    return None


def build_game_state(bot_json: dict[str, Any]) -> dict[str, Any]:
    """Convert a balatrobot gamestate response to an engine game_state dict.

    Unlike jackdaw's ``bot_state_to_game_state()`` which only extracts keys,
    this builds real Card objects suitable for ``encode_observation()``.

    Args:
        bot_json: The full gamestate dict from ``BalatroClient.gamestate()``.

    Returns:
        Engine game_state dict compatible with ``encode_observation()``.
    """
    gs: dict[str, Any] = {}

    # Phase
    state_str = bot_json.get("state", "GAME_OVER")
    gs["phase"] = _STATE_MAP.get(state_str, GamePhase.GAME_OVER)

    # Economy
    gs["dollars"] = bot_json.get("money", 0)

    # Ante / round
    gs["round_resets"] = {
        "ante": bot_json.get("ante_num", 1),
        "blind_states": {},
        "blind_choices": {},
        "blind_tags": {},
    }
    gs["round"] = bot_json.get("round_num", 0)

    # Blind states from blinds JSON
    blinds_json = bot_json.get("blinds", {})
    _status_from_bot = {
        "SELECT": "Select",
        "CURRENT": "Current",
        "SKIPPED": "Skipped",
        "DEFEATED": "Defeated",
        "UPCOMING": "Upcoming",
    }
    for btype_lower, btype_cap in [("small", "Small"), ("big", "Big"), ("boss", "Boss")]:
        bi = blinds_json.get(btype_lower, {})
        status_bot = bi.get("status", "UPCOMING")
        gs["round_resets"]["blind_states"][btype_cap] = _status_from_bot.get(status_bot, status_bot)

    # Determine blind_on_deck from blind states
    gs["blind_on_deck"] = None
    for btype_cap in ("Small", "Big", "Boss"):
        state = gs["round_resets"]["blind_states"].get(btype_cap, "")
        if state in ("Select", "Current"):
            gs["blind_on_deck"] = btype_cap
            break

    # Round state
    br = bot_json.get("round", {})
    gs["current_round"] = {
        "hands_left": br.get("hands_left", 0),
        "discards_left": br.get("discards_left", 0),
        "hands_played": br.get("hands_played", 0),
        "discards_used": br.get("discards_used", 0),
        "reroll_cost": br.get("reroll_cost", 5),
        "free_rerolls": 0,
    }
    gs["chips"] = br.get("chips", 0)

    # Cards — build real Card objects
    hand_area = bot_json.get("hand", {})
    gs["hand"] = _build_cards_from_area(hand_area)
    gs["hand_size"] = hand_area.get("limit", 8)

    deck_area = bot_json.get("cards", {})
    gs["deck"] = _build_cards_from_area(deck_area)

    joker_area = bot_json.get("jokers", {})
    gs["jokers"] = _build_cards_from_area(joker_area)
    gs["joker_slots"] = joker_area.get("limit", 5)

    consumable_area = bot_json.get("consumables", {})
    gs["consumables"] = _build_cards_from_area(consumable_area)
    gs["consumable_slots"] = consumable_area.get("limit", 2)

    # Shop
    shop_area = bot_json.get("shop", {})
    gs["shop_cards"] = _build_cards_from_area(shop_area)

    voucher_area = bot_json.get("vouchers", {})
    gs["shop_vouchers"] = _build_cards_from_area(voucher_area)

    pack_area = bot_json.get("packs", {})
    gs["shop_boosters"] = _build_cards_from_area(pack_area)

    # Pack opening
    open_pack_area = bot_json.get("pack", {})
    gs["pack_cards"] = _build_cards_from_area(open_pack_area)
    gs["pack_choices_remaining"] = open_pack_area.get("limit", 0)

    # Hand levels
    gs["hand_levels"] = _build_hand_levels(bot_json.get("hands", {}))

    # Blind object
    gs["blind"] = _build_blind(blinds_json, gs)

    # Used vouchers
    gs["used_vouchers"] = bot_json.get("used_vouchers", {})

    # Discard pile (not in balatrobot JSON — empty)
    gs["discard_pile"] = []

    # Misc scalars
    gs["skips"] = 0
    gs["interest_cap"] = 25
    gs["discount_percent"] = 0
    gs["won"] = bot_json.get("won", False)
    gs["seed"] = bot_json.get("seed", "")
    gs["four_fingers"] = 0
    gs["shortcut"] = 0
    gs["smeared"] = 0
    gs["splash"] = 0
    gs["awarded_tags"] = []

    return gs
