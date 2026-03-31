"""Decision tracer for heuristic agents.

Wraps a PhaseDispatchAgent and logs human-readable decision traces to stderr.
Useful for debugging heuristic logic and understanding agent behavior.

Usage::

    agent = PhaseDispatchAgent(...)
    traced = DecisionTracer(agent)
    action = traced.select_action(obs, mask, action_table=table)
"""

from __future__ import annotations

import math
import sys
from typing import TextIO

import numpy as np

from balatro_rl.agents.dispatch import Phase, PhaseDispatchAgent, _detect_phase
from balatro_rl.features.hand_evaluator import (
    evaluate_hand_from_parsed,
    parse_cards_from_obs,
    parse_jokers_from_obs,
)
from balatro_rl.features.joker_catalog import (
    decode_shop_card_set,
    decode_shop_center_id,
    id_to_key,
)
from balatro_rl.features.joker_scoring import GameContext, simulate_joker_scoring

_ACTION_TYPE_NAMES = [
    "PlayHand", "Discard", "SelectBlind", "SkipBlind", "CashOut",
    "Reroll", "NextRound", "SkipPack", "BuyCard", "SellJoker",
    "SellConsumable", "UseConsumable", "RedeemVoucher", "OpenBooster",
    "PickPackCard", "SwapJokersLeft", "SwapJokersRight", "SwapHandLeft",
    "SwapHandRight", "SortHandRank", "SortHandSuit",
]

_BLIND_NAMES = {0: "???", 1: "Small", 2: "Big", 3: "Boss"}


def _inv_log_scale(v: float) -> float:
    """Invert jackdaw's log_scale: sign(x) * log2(1 + |x|)."""
    if v >= 0:
        return 2.0 ** v - 1.0
    return -(2.0 ** (-v) - 1.0)


def _card_str(rank: str, suit: str) -> str:
    suit_symbol = {"H": "h", "D": "d", "C": "c", "S": "s"}.get(suit, "?")
    return f"{rank}{suit_symbol}"


class DecisionTracer:
    """Wraps a dispatch agent and prints decisions to a stream."""

    def __init__(
        self,
        agent: PhaseDispatchAgent,
        out: TextIO = sys.stderr,
        hand_detail: bool = True,
        shop_detail: bool = True,
    ) -> None:
        self.agent = agent
        self.out = out
        self.hand_detail = hand_detail
        self.shop_detail = shop_detail
        self._step = 0
        self._prev_phase: Phase | None = None
        self._episode = 0
        self._jokers_logged = False

    def new_episode(self, episode: int = 0) -> None:
        self._step = 0
        self._prev_phase = None
        self._jokers_logged = False
        self._episode = episode
        self._print(f"\n{'='*60}")
        self._print(f"EPISODE {episode}")
        self._print(f"{'='*60}")

    def end_episode(self, info: dict) -> None:
        ante = info.get("balatro/ante_reached", "?")
        won = info.get("balatro/won", False)
        self._print(f"--- END: ante_reached={ante}, won={won}")

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        action_mask: np.ndarray,
        action_table: list | None = None,
    ) -> int:
        phase = _detect_phase(obs)
        g = obs["global"]
        ante = max(round(g[10] * 8.0), 1)
        blind_type = _BLIND_NAMES.get(round(g[6] + g[7] * 2 + g[8] * 3 + g[9] * 4), "???")

        # Decode round position: global[130:133] one-hot (small/big/boss)
        round_pos = int(np.argmax(g[130:133]))
        round_name = ["Small", "Big", "Boss"][round_pos] if g[130:133].any() else blind_type

        # Print phase transitions
        if phase != self._prev_phase:
            blind_chips = _inv_log_scale(g[18])
            chips_scored = _inv_log_scale(g[19])
            dollars = _inv_log_scale(g[12])
            hands_left = round(g[13] * 10.0)
            discards_left = round(g[14] * 10.0)

            if phase == Phase.SELECTING_HAND:
                self._jokers_logged = False
                self._print(
                    f"\n[Ante {ante} {round_name}] HAND PLAY "
                    f"| target={blind_chips:.0f} scored={chips_scored:.0f} "
                    f"| hands={hands_left} discards={discards_left} | ${dollars:.0f}"
                )
            elif phase == Phase.SHOP:
                self._print(f"\n[Ante {ante}] SHOP | ${dollars:.0f}")
            elif phase == Phase.BLIND_SELECT:
                self._print(f"\n[Ante {ante}] BLIND SELECT")
            elif phase == Phase.PACK_OPENING:
                self._print(f"  [PACK OPENING]")
            self._prev_phase = phase

        action = self.agent.select_action(obs, action_mask, action_table=action_table)

        # Decode and print the chosen action
        if action_table is not None and action < len(action_table):
            fa = action_table[action]
            action_name = _ACTION_TYPE_NAMES[fa.action_type] if fa.action_type < len(_ACTION_TYPE_NAMES) else f"type{fa.action_type}"

            if phase == Phase.SELECTING_HAND and self.hand_detail:
                self._trace_hand_action(obs, fa, action_name)
            elif phase == Phase.SHOP and self.shop_detail:
                self._trace_shop_action(obs, fa, action_name)
            elif phase == Phase.BLIND_SELECT:
                self._print(f"  -> {action_name}")
            elif phase == Phase.PACK_OPENING:
                self._trace_pack_action(obs, fa, action_name)
            else:
                self._print(f"  -> {action_name}")

        self._step += 1
        return action

    def _trace_hand_action(self, obs: dict, fa: object, action_name: str) -> None:
        g = obs["global"]
        n_cards = int(obs["entity_counts"][0])
        cards = parse_cards_from_obs(obs["hand_card"], n_cards)
        hand_str = " ".join(_card_str(c.rank, c.suit) for c in cards)

        n_jokers = int(obs["entity_counts"][1])
        jokers = parse_jokers_from_obs(obs["joker"], n_jokers)

        # Log jokers on first hand action of a new round
        if jokers and not self._jokers_logged:
            joker_names = [j.key.replace("j_", "") for j in jokers]
            self._print(f"  jokers: [{', '.join(joker_names)}]")
            self._jokers_logged = True

        if fa.action_type == 0:  # PlayHand
            played = [cards[i] for i in fa.card_target if i < len(cards)] if fa.card_target else []
            played_str = " ".join(_card_str(c.rank, c.suit) for c in played)
            candidates = evaluate_hand_from_parsed(cards, max_candidates=3)
            best = candidates[0] if candidates else None

            if best and jokers:
                discards_left = round(g[14] * 10.0)
                deck_size = round(_inv_log_scale(g[27]))
                ctx = GameContext(
                    discards_left=discards_left,
                    n_jokers=n_jokers,
                    deck_size=max(deck_size, 0),
                    hand_cards=cards,
                )
                joker_score = simulate_joker_scoring(best, jokers, ctx)
                est = (
                    f" base={best.estimated_chips:.0f}"
                    f" joker={joker_score:.0f} ({best.hand_type})"
                )
            elif best:
                est = f" est={best.estimated_chips:.0f} ({best.hand_type})"
            else:
                est = ""
            self._print(f"  hand: [{hand_str}]")
            self._print(f"  -> PLAY [{played_str}]{est}")
        elif fa.action_type == 1:  # Discard
            discarded = [cards[i] for i in fa.card_target if i < len(cards)] if fa.card_target else []
            disc_str = " ".join(_card_str(c.rank, c.suit) for c in discarded)
            kept = [c for c in cards if c.index not in (fa.card_target or ())]
            kept_str = " ".join(_card_str(c.rank, c.suit) for c in kept)
            self._print(f"  hand: [{hand_str}]")
            self._print(f"  -> DISCARD [{disc_str}] keep [{kept_str}]")

    def _trace_shop_action(self, obs: dict, fa: object, action_name: str) -> None:
        n_shop = int(obs["entity_counts"][3])
        shop_items = obs["shop_item"]

        if fa.action_type in (8, 13) and fa.entity_target is not None:  # BuyCard / OpenBooster
            slot = fa.entity_target
            if slot < n_shop:
                row = shop_items[slot]
                center_id = decode_shop_center_id(row[0])
                card_set = decode_shop_card_set(row[1])
                key = id_to_key(center_id)
                cost = _inv_log_scale(row[2])
                name = key.replace("j_", "").replace("c_", "").replace("p_", "").replace("_", " ").title()
                self._print(f"  -> {action_name}: {name} ({card_set}, {key}) ${cost:.0f}")
            else:
                self._print(f"  -> {action_name}: slot {slot}")
        elif fa.action_type in (6, 4):  # NextRound / CashOut
            dollars = _inv_log_scale(obs["global"][12])
            self._print(f"  -> {action_name} (${dollars:.0f})")
        else:
            self._print(f"  -> {action_name}")

    def _trace_pack_action(self, obs: dict, fa: object, action_name: str) -> None:
        if fa.action_type == 14 and fa.entity_target is not None:  # PickPackCard
            n_pack = int(obs["entity_counts"][4])
            if fa.entity_target < n_pack:
                pack_cards = obs["pack_card"]
                row = pack_cards[fa.entity_target]
                rank_id = round(row[0] * 14)
                suit_id = round(row[1] * 3)
                rank = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
                        9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}.get(rank_id, "?")
                suit = {0: "H", 1: "D", 2: "C", 3: "S"}.get(suit_id, "?")
                self._print(f"  -> PickPackCard: {_card_str(rank, suit)}")
            else:
                self._print(f"  -> PickPackCard: idx {fa.entity_target}")
        elif fa.action_type == 7:  # SkipPack
            self._print(f"  -> SkipPack")
        else:
            self._print(f"  -> {action_name}")

    def _print(self, msg: str) -> None:
        print(msg, file=self.out)

    def update(self, **kwargs: object) -> None:
        self.agent.update(**kwargs)
