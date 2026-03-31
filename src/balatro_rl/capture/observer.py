"""Passive gamestate observer for expert gameplay capture.

Polls ``BalatroClient.gamestate()`` at a configurable interval, detects
state transitions, infers actions from consecutive state diffs, and
feeds (state, action) pairs to a SessionRecorder.

Action inference uses card identity (``id`` field), phase transitions,
and economic signals (money changes, chip changes) to reconstruct
what the player did.

Usage::

    client = BalatroClient()
    recorder = SessionRecorder(save_dir="data/captures")
    observer = GameObserver(client, recorder)
    observer.run()  # blocks until game ends or Ctrl-C
"""

from __future__ import annotations

import time
from typing import Any

from balatro_rl.capture.recorder import SessionRecorder
from balatro_rl.client import BalatroClient


def infer_action(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any] | None:
    """Infer the action taken between two consecutive gamestates.

    Returns a dict ``{"method": str, "params": dict}`` matching the
    balatrobot JSON-RPC format, or None if no action can be inferred
    (e.g., states are identical).
    """
    prev_state = prev.get("state", "")
    curr_state = curr.get("state", "")
    prev_round = prev.get("round", {})
    curr_round = curr.get("round", {})

    # Phase transitions with unambiguous action mapping
    if prev_state == "BLIND_SELECT" and curr_state == "SELECTING_HAND":
        return {"method": "select", "params": {}}

    if prev_state == "BLIND_SELECT" and curr_state in ("BLIND_SELECT", "SHOP"):
        if prev.get("ante_num") == curr.get("ante_num"):
            return {"method": "skip", "params": {}}

    if prev_state == "ROUND_EVAL" and curr_state == "SHOP":
        return {"method": "cash_out", "params": {}}

    if prev_state == "SHOP" and curr_state == "BLIND_SELECT":
        return {"method": "next_round", "params": {}}

    # Hand phase: play or discard
    if prev_state == "SELECTING_HAND" and curr_state in ("SELECTING_HAND", "ROUND_EVAL"):
        prev_hand_ids = _card_ids(prev.get("hand", {}))
        curr_hand_ids = _card_ids(curr.get("hand", {}))
        removed_ids = prev_hand_ids - curr_hand_ids

        if not removed_ids:
            return None

        prev_cards = prev.get("hand", {}).get("cards", [])
        removed_indices = _ids_to_indices(prev_cards, removed_ids)

        prev_chips = prev_round.get("chips", 0)
        curr_chips = curr_round.get("chips", 0)
        prev_hands = prev_round.get("hands_left", 0)
        curr_hands = curr_round.get("hands_left", 0)

        if curr_chips > prev_chips or curr_hands < prev_hands:
            return {"method": "play", "params": {"cards": removed_indices}}
        else:
            return {"method": "discard", "params": {"cards": removed_indices}}

    # Shop actions
    if prev_state == "SHOP" and curr_state == "SHOP":
        return _infer_shop_action(prev, curr)

    # Pack opening
    if prev_state == "SMODS_BOOSTER_OPENED":
        return _infer_pack_action(prev, curr)

    # SHOP -> SMODS_BOOSTER_OPENED (opened a pack)
    if prev_state == "SHOP" and curr_state == "SMODS_BOOSTER_OPENED":
        prev_packs = prev.get("packs", {}).get("cards", [])
        curr_packs = curr.get("packs", {}).get("cards", [])
        if len(curr_packs) < len(prev_packs):
            prev_ids = {c.get("id") for c in prev_packs}
            curr_ids = {c.get("id") for c in curr_packs}
            removed = prev_ids - curr_ids
            for i, c in enumerate(prev_packs):
                if c.get("id") in removed:
                    return {"method": "buy", "params": {"pack": i}}
        return {"method": "buy", "params": {"pack": 0}}

    return None


def _infer_shop_action(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any] | None:
    """Infer a shop-phase action from two consecutive shop states."""
    prev_money = prev.get("money", 0)
    curr_money = curr.get("money", 0)

    prev_shop_ids = _card_ids(prev.get("shop", {}))
    curr_shop_ids = _card_ids(curr.get("shop", {}))
    removed_shop = prev_shop_ids - curr_shop_ids
    added_shop = curr_shop_ids - prev_shop_ids

    # Reroll: ALL shop cards replaced (no overlap between old and new IDs),
    # money decreased. Check before buy since buy also has removed cards.
    if (removed_shop and added_shop
            and not (prev_shop_ids & curr_shop_ids)
            and curr_money < prev_money):
        return {"method": "reroll", "params": {}}

    # Bought a shop card (specific card disappeared, money decreased,
    # remaining cards retained)
    if removed_shop and curr_money < prev_money:
        prev_cards = prev.get("shop", {}).get("cards", [])
        for i, c in enumerate(prev_cards):
            if c.get("id") in removed_shop:
                return {"method": "buy", "params": {"card": i}}

    # Sold a joker (joker disappeared, money increased)
    prev_joker_ids = _card_ids(prev.get("jokers", {}))
    curr_joker_ids = _card_ids(curr.get("jokers", {}))
    removed_jokers = prev_joker_ids - curr_joker_ids
    if removed_jokers and curr_money > prev_money:
        prev_jokers = prev.get("jokers", {}).get("cards", [])
        for i, c in enumerate(prev_jokers):
            if c.get("id") in removed_jokers:
                return {"method": "sell", "params": {"joker": i}}

    # Sold a consumable
    prev_cons_ids = _card_ids(prev.get("consumables", {}))
    curr_cons_ids = _card_ids(curr.get("consumables", {}))
    removed_cons = prev_cons_ids - curr_cons_ids
    if removed_cons and curr_money > prev_money:
        prev_cons = prev.get("consumables", {}).get("cards", [])
        for i, c in enumerate(prev_cons):
            if c.get("id") in removed_cons:
                return {"method": "sell", "params": {"consumable": i}}

    # Bought a voucher
    prev_voucher_ids = _card_ids(prev.get("vouchers", {}))
    curr_voucher_ids = _card_ids(curr.get("vouchers", {}))
    if prev_voucher_ids - curr_voucher_ids and curr_money < prev_money:
        prev_vouchers = prev.get("vouchers", {}).get("cards", [])
        removed_vouchers = prev_voucher_ids - curr_voucher_ids
        for i, c in enumerate(prev_vouchers):
            if c.get("id") in removed_vouchers:
                return {"method": "buy", "params": {"voucher": i}}

    return None


def _infer_pack_action(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any] | None:
    """Infer a pack-opening action from two consecutive states."""
    curr_state = curr.get("state", "")

    prev_pack_ids = _card_ids(prev.get("pack", {}))
    curr_pack_ids = _card_ids(curr.get("pack", {}))
    removed = prev_pack_ids - curr_pack_ids

    # Pack closed (left SMODS_BOOSTER_OPENED): if all cards removed, it's a skip
    if curr_state != "SMODS_BOOSTER_OPENED":
        if removed == prev_pack_ids and prev_pack_ids:
            return {"method": "pack", "params": {"skip": True}}
        if not prev_pack_ids:
            return {"method": "pack", "params": {"skip": True}}

    # Single card removed from pack while still open: pick
    if removed and curr_state == "SMODS_BOOSTER_OPENED":
        prev_cards = prev.get("pack", {}).get("cards", [])
        for i, c in enumerate(prev_cards):
            if c.get("id") in removed:
                return {"method": "pack", "params": {"card": i}}

    # Remaining cards but one was picked and we transitioned out
    if removed and len(removed) < len(prev_pack_ids):
        prev_cards = prev.get("pack", {}).get("cards", [])
        for i, c in enumerate(prev_cards):
            if c.get("id") in removed:
                return {"method": "pack", "params": {"card": i}}

    return None


def _card_ids(area: dict[str, Any]) -> set[int]:
    """Extract the set of card ids from an area JSON."""
    return {c.get("id") for c in area.get("cards", []) if "id" in c}


def _ids_to_indices(cards: list[dict], target_ids: set[int]) -> list[int]:
    """Map card ids to their 0-based indices in the card list."""
    return sorted(i for i, c in enumerate(cards) if c.get("id") in target_ids)


class GameObserver:
    """Passive observer that polls balatrobot and records transitions.

    Args:
        client: Connected BalatroClient instance.
        recorder: SessionRecorder to write transitions to.
        poll_interval: Seconds between gamestate polls.
    """

    def __init__(
        self,
        client: BalatroClient,
        recorder: SessionRecorder,
        poll_interval: float = 0.2,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self.poll_interval = poll_interval
        self._prev_state: dict[str, Any] | None = None
        self._running = False

    def run(self, scenario: str | None = None) -> dict[str, Any]:
        """Run the observer loop until the game ends or interrupted.

        Returns a summary dict with session stats.
        """
        self._running = True
        self.recorder.begin_session(mode="observe", scenario=scenario)

        transitions = 0
        try:
            while self._running:
                try:
                    curr = self.client.gamestate()
                except Exception:
                    time.sleep(self.poll_interval)
                    continue

                if self._prev_state is None:
                    self.recorder.record_transition(curr)
                    self._prev_state = curr
                    time.sleep(self.poll_interval)
                    continue

                if _states_differ(self._prev_state, curr):
                    action = infer_action(self._prev_state, curr)
                    self.recorder.record_transition(
                        self._prev_state,
                        action=action,
                        inferred=True,
                    )
                    transitions += 1
                    self._prev_state = curr

                    if curr.get("state") == "GAME_OVER":
                        self.recorder.record_transition(curr)
                        break

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            pass

        ante = 0
        won = False
        if self._prev_state:
            ante = self._prev_state.get("ante_num", 0)
            won = self._prev_state.get("won", False)

        path = self.recorder.end_session(ante_reached=ante, won=won)
        self._prev_state = None

        return {
            "path": str(path),
            "transitions": transitions,
            "ante_reached": ante,
            "won": won,
        }

    def stop(self) -> None:
        """Signal the observer to stop after the current poll."""
        self._running = False


def _states_differ(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Check if two gamestates represent a meaningful change."""
    if a.get("state") != b.get("state"):
        return True

    ar = a.get("round", {})
    br = b.get("round", {})
    if ar.get("hands_left") != br.get("hands_left"):
        return True
    if ar.get("discards_left") != br.get("discards_left"):
        return True
    if ar.get("chips") != br.get("chips"):
        return True

    if a.get("money") != b.get("money"):
        return True
    if a.get("ante_num") != b.get("ante_num"):
        return True

    if _card_ids(a.get("hand", {})) != _card_ids(b.get("hand", {})):
        return True
    if _card_ids(a.get("jokers", {})) != _card_ids(b.get("jokers", {})):
        return True
    if _card_ids(a.get("shop", {})) != _card_ids(b.get("shop", {})):
        return True
    if _card_ids(a.get("pack", {})) != _card_ids(b.get("pack", {})):
        return True
    if _card_ids(a.get("consumables", {})) != _card_ids(b.get("consumables", {})):
        return True

    return False
