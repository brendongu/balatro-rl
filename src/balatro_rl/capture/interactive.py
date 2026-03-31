"""Interactive terminal harness for API-driven expert gameplay capture.

Presents the current game state and legal actions in a human-readable
format. The expert selects actions by number, which are sent to the
live game via BalatroClient. All transitions are recorded with exact
action labels (no inference needed).

Usage::

    client = BalatroClient()
    recorder = SessionRecorder(save_dir="data/captures")
    harness = InteractiveHarness(client, recorder)
    harness.run()
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from balatro_rl.capture.recorder import SessionRecorder
from balatro_rl.client import BalatroClient

_RANK_DISPLAY: dict[str, str] = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "10",
    "J": "J", "Q": "Q", "K": "K", "A": "A",
}
_SUIT_SYMBOL: dict[str, str] = {"H": "h", "D": "d", "C": "c", "S": "s"}


def _card_str(card: dict[str, Any]) -> str:
    """Format a balatrobot card JSON as a short human-readable string."""
    card_set = card.get("set", "DEFAULT")
    value = card.get("value", {})

    if card_set in ("JOKER", "TAROT", "PLANET", "SPECTRAL", "VOUCHER", "BOOSTER"):
        label = card.get("label", card.get("key", "?"))
        modifier = card.get("modifier", {})
        parts = [label]
        if modifier.get("edition"):
            parts.append(f"[{modifier['edition']}]")
        if modifier.get("eternal"):
            parts.append("[Eternal]")
        cost = card.get("cost", {}).get("buy", 0)
        parts.append(f"(${cost})")
        return " ".join(parts)

    rank = _RANK_DISPLAY.get(value.get("rank", "?"), "?")
    suit = _SUIT_SYMBOL.get(value.get("suit", "?"), "?")
    base = f"{rank}{suit}"

    modifier = card.get("modifier", {})
    extras = []
    if modifier.get("enhancement"):
        extras.append(modifier["enhancement"])
    if modifier.get("edition"):
        extras.append(modifier["edition"])
    if modifier.get("seal"):
        extras.append(f"{modifier['seal']} Seal")
    if card.get("state", {}).get("debuff"):
        extras.append("DEBUFF")

    if extras:
        return f"{base}({','.join(extras)})"
    return base


def _display_state(gs: dict[str, Any], out: TextIO) -> None:
    """Print a human-readable summary of the current gamestate."""
    state = gs.get("state", "?")
    ante = gs.get("ante_num", 1)
    money = gs.get("money", 0)
    rnd = gs.get("round", {})

    out.write(f"\n{'='*60}\n")
    out.write(f"  Phase: {state}  |  Ante: {ante}  |  ${money}\n")

    if state in ("SELECTING_HAND", "ROUND_EVAL"):
        chips = rnd.get("chips", 0)
        hands = rnd.get("hands_left", 0)
        discards = rnd.get("discards_left", 0)

        blinds = gs.get("blinds", {})
        target = 0
        for bt in ("small", "big", "boss"):
            bi = blinds.get(bt, {})
            if bi.get("status") == "CURRENT":
                target = bi.get("score", 0)
                out.write(f"  Blind: {bi.get('name', bt)} ({target} chips)\n")
                break

        out.write(f"  Chips: {chips}/{target}  |  Hands: {hands}  |  Discards: {discards}\n")

    # Hand
    hand_cards = gs.get("hand", {}).get("cards", [])
    if hand_cards:
        card_strs = [f"[{i}]{_card_str(c)}" for i, c in enumerate(hand_cards)]
        out.write(f"  Hand: {' '.join(card_strs)}\n")

    # Jokers
    joker_cards = gs.get("jokers", {}).get("cards", [])
    if joker_cards:
        joker_strs = [_card_str(c) for c in joker_cards]
        out.write(f"  Jokers: {', '.join(joker_strs)}\n")

    # Consumables
    cons_cards = gs.get("consumables", {}).get("cards", [])
    if cons_cards:
        cons_strs = [_card_str(c) for c in cons_cards]
        out.write(f"  Consumables: {', '.join(cons_strs)}\n")

    # Shop
    if state == "SHOP":
        shop_cards = gs.get("shop", {}).get("cards", [])
        if shop_cards:
            shop_strs = [f"[{i}]{_card_str(c)}" for i, c in enumerate(shop_cards)]
            out.write(f"  Shop: {' '.join(shop_strs)}\n")

        vouchers = gs.get("vouchers", {}).get("cards", [])
        if vouchers:
            v_strs = [f"[{i}]{_card_str(c)}" for i, c in enumerate(vouchers)]
            out.write(f"  Vouchers: {' '.join(v_strs)}\n")

        packs = gs.get("packs", {}).get("cards", [])
        if packs:
            p_strs = [f"[{i}]{_card_str(c)}" for i, c in enumerate(packs)]
            out.write(f"  Packs: {' '.join(p_strs)}\n")

        reroll_cost = rnd.get("reroll_cost", 5)
        out.write(f"  Reroll cost: ${reroll_cost}\n")

    # Pack opening
    if state == "SMODS_BOOSTER_OPENED":
        pack_cards = gs.get("pack", {}).get("cards", [])
        if pack_cards:
            pk_strs = [f"[{i}]{_card_str(c)}" for i, c in enumerate(pack_cards)]
            out.write(f"  Pack cards: {' '.join(pk_strs)}\n")

    # Blind select
    if state == "BLIND_SELECT":
        blinds = gs.get("blinds", {})
        for bt in ("small", "big", "boss"):
            bi = blinds.get(bt, {})
            if bi.get("status") in ("SELECT", "UPCOMING"):
                out.write(f"  {bt.title()}: {bi.get('name', '?')} - {bi.get('score', 0)} chips")
                if bi.get("tag_name"):
                    out.write(f" [Tag: {bi['tag_name']}]")
                out.write("\n")

    out.write(f"{'='*60}\n")


def _build_action_menu(gs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a list of available actions based on the current state."""
    state = gs.get("state", "")
    actions: list[dict[str, Any]] = []

    if state == "BLIND_SELECT":
        actions.append({"label": "Select blind", "method": "select", "params": {}})
        actions.append({"label": "Skip blind", "method": "skip", "params": {}})

    elif state == "SELECTING_HAND":
        hand_cards = gs.get("hand", {}).get("cards", [])
        n = len(hand_cards)

        actions.append({
            "label": f"Play cards (enter indices 0-{n-1}, e.g. '0 2 4')",
            "method": "play",
            "params": "prompt_cards",
        })
        actions.append({
            "label": f"Discard cards (enter indices 0-{n-1})",
            "method": "discard",
            "params": "prompt_cards",
        })

        cons = gs.get("consumables", {}).get("cards", [])
        for i, c in enumerate(cons):
            label = _card_str(c)
            actions.append({
                "label": f"Use consumable {i}: {label}",
                "method": "use",
                "params": {"consumable": i},
            })

    elif state == "ROUND_EVAL":
        actions.append({"label": "Cash out", "method": "cash_out", "params": {}})

    elif state == "SHOP":
        shop_cards = gs.get("shop", {}).get("cards", [])
        for i, c in enumerate(shop_cards):
            cost = c.get("cost", {}).get("buy", 0)
            actions.append({
                "label": f"Buy card {i}: {_card_str(c)} (${cost})",
                "method": "buy",
                "params": {"card": i},
            })

        vouchers = gs.get("vouchers", {}).get("cards", [])
        for i, c in enumerate(vouchers):
            cost = c.get("cost", {}).get("buy", 0)
            actions.append({
                "label": f"Buy voucher {i}: {_card_str(c)} (${cost})",
                "method": "buy",
                "params": {"voucher": i},
            })

        packs = gs.get("packs", {}).get("cards", [])
        for i, c in enumerate(packs):
            cost = c.get("cost", {}).get("buy", 0)
            actions.append({
                "label": f"Open pack {i}: {_card_str(c)} (${cost})",
                "method": "buy",
                "params": {"pack": i},
            })

        jokers = gs.get("jokers", {}).get("cards", [])
        for i, c in enumerate(jokers):
            sell = c.get("cost", {}).get("sell", 0)
            actions.append({
                "label": f"Sell joker {i}: {_card_str(c)} (+${sell})",
                "method": "sell",
                "params": {"joker": i},
            })

        cons = gs.get("consumables", {}).get("cards", [])
        for i, c in enumerate(cons):
            sell = c.get("cost", {}).get("sell", 0)
            actions.append({
                "label": f"Sell consumable {i}: {_card_str(c)} (+${sell})",
                "method": "sell",
                "params": {"consumable": i},
            })

        reroll_cost = gs.get("round", {}).get("reroll_cost", 5)
        actions.append({"label": f"Reroll (${reroll_cost})", "method": "reroll", "params": {}})
        actions.append({"label": "Next round", "method": "next_round", "params": {}})

    elif state == "SMODS_BOOSTER_OPENED":
        pack_cards = gs.get("pack", {}).get("cards", [])
        for i, c in enumerate(pack_cards):
            actions.append({
                "label": f"Pick card {i}: {_card_str(c)}",
                "method": "pack",
                "params": {"card": i},
            })
        actions.append({"label": "Skip pack", "method": "pack", "params": {"skip": True}})

    return actions


def _execute_action(
    client: BalatroClient,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Execute an action via BalatroClient and return the new gamestate."""
    client._call(method, params if params else None)
    return client.gamestate()


def _prompt_card_indices(prompt: str, out: TextIO) -> list[int]:
    """Prompt the user for card indices."""
    out.write(f"  {prompt}: ")
    out.flush()
    line = input().strip()
    if not line:
        return []
    return [int(x) for x in line.split()]


class InteractiveHarness:
    """Terminal-based interactive capture harness.

    Args:
        client: Connected BalatroClient.
        recorder: SessionRecorder to write transitions to.
        out: Output stream for display.
    """

    def __init__(
        self,
        client: BalatroClient,
        recorder: SessionRecorder,
        out: TextIO = sys.stdout,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self.out = out

    def run(self, scenario: str | None = None) -> dict[str, Any]:
        """Run the interactive capture loop until the game ends or 'q' is entered.

        Returns a summary dict with session stats.
        """
        self.recorder.begin_session(mode="interactive", scenario=scenario)
        transitions = 0

        try:
            gs = self.client.gamestate()

            while gs.get("state") != "GAME_OVER":
                _display_state(gs, self.out)
                actions = _build_action_menu(gs)

                if not actions:
                    self.out.write("  No actions available. Waiting...\n")
                    import time
                    time.sleep(1)
                    gs = self.client.gamestate()
                    continue

                self.out.write("\n  Actions:\n")
                for i, a in enumerate(actions):
                    self.out.write(f"    {i}: {a['label']}\n")
                self.out.write("    q: Quit\n")

                self.out.write("\n  Choice: ")
                self.out.flush()
                choice = input().strip()

                if choice.lower() == "q":
                    break

                try:
                    idx = int(choice)
                except ValueError:
                    self.out.write("  Invalid choice.\n")
                    continue

                if idx < 0 or idx >= len(actions):
                    self.out.write("  Out of range.\n")
                    continue

                selected = actions[idx]
                method = selected["method"]
                params = selected["params"]

                if params == "prompt_cards":
                    indices = _prompt_card_indices("Enter card indices (space-separated)", self.out)
                    if not indices:
                        self.out.write("  No cards selected.\n")
                        continue
                    params = {"cards": indices}

                action_record = {"method": method, "params": params}
                self.recorder.record_transition(gs, action=action_record, inferred=False)
                transitions += 1

                try:
                    gs = _execute_action(self.client, method, params)
                except Exception as e:
                    self.out.write(f"  Error: {e}\n")
                    gs = self.client.gamestate()

            # Record terminal state
            self.recorder.record_transition(gs)

        except KeyboardInterrupt:
            pass

        ante = gs.get("ante_num", 0) if gs else 0
        won = gs.get("won", False) if gs else False

        path = self.recorder.end_session(ante_reached=ante, won=won)

        return {
            "path": str(path),
            "transitions": transitions,
            "ante_reached": ante,
            "won": won,
        }
