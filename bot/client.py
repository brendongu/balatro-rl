"""
Thin wrapper around the balatrobot JSON-RPC 2.0 API.

Provides one method per API endpoint, typed inputs/outputs, and clean
error handling. Everything returns the raw gamestate dict so callers
can parse it however they need (env.py, scripts, etc.).

API endpoint: http://127.0.0.1:12346 (default)
"""

from __future__ import annotations

import requests
from typing import Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BalatroError(Exception):
    """Base exception for all balatrobot API errors."""

    def __init__(self, code: int, message: str, name: str = ""):
        self.code = code
        self.name = name
        super().__init__(f"[{name or code}] {message}")


class BadRequestError(BalatroError):
    """Invalid parameters or protocol error (-32001)."""


class InvalidStateError(BalatroError):
    """Action not allowed in current game state (-32002)."""


class NotAllowedError(BalatroError):
    """Game rules prevent this action (-32003)."""


class InternalError(BalatroError):
    """Server-side failure (-32000)."""


_ERROR_MAP = {
    -32000: InternalError,
    -32001: BadRequestError,
    -32002: InvalidStateError,
    -32003: NotAllowedError,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class BalatroClient:
    """
    JSON-RPC 2.0 client for the balatrobot API.

    Usage:
        client = BalatroClient()
        client.health()
        state = client.start(deck="RED", stake="WHITE")
        state = client.select()
        state = client.play(cards=[0, 1, 2])
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 12346, timeout: int = 10):
        self.url = f"http://{host}:{port}"
        self.timeout = timeout
        self._id = 0

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _call(self, method: str, params: Optional[dict] = None) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
        }
        if params:
            payload["params"] = params

        try:
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to balatrobot at {self.url}. "
                "Is Balatro running with the balatrobot mod?"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"balatrobot API timed out after {self.timeout}s.")

        data = response.json()

        if "error" in data:
            err = data["error"]
            code = err.get("code", 0)
            message = err.get("message", "Unknown error")
            name = (err.get("data") or {}).get("name", "")
            exc_class = _ERROR_MAP.get(code, BalatroError)
            raise exc_class(code, message, name)

        return data["result"]

    # -----------------------------------------------------------------------
    # Health & State
    # -----------------------------------------------------------------------

    def health(self) -> dict:
        """Health check. Returns {"status": "ok"} if server is up."""
        return self._call("health")

    def gamestate(self) -> dict:
        """Get the complete current game state."""
        return self._call("gamestate")

    def discover(self) -> dict:
        """Returns the OpenRPC specification."""
        return self._call("rpc.discover")

    # -----------------------------------------------------------------------
    # Game Flow
    # -----------------------------------------------------------------------

    def start(
        self,
        deck: str = "RED",
        stake: str = "WHITE",
        seed: Optional[str] = None,
    ) -> dict:
        """
        Start a new run. Returns gamestate (state=BLIND_SELECT).

        Args:
            deck:  Deck type. One of: RED, BLUE, YELLOW, GREEN, BLACK,
                   MAGIC, NEBULA, GHOST, ABANDONED, CHECKERED, ZODIAC,
                   PAINTED, ANAGLYPH, PLASMA, ERRATIC
            stake: Stake level. One of: WHITE, RED, GREEN, BLACK, BLUE,
                   PURPLE, ORANGE, GOLD
            seed:  Optional seed string for reproducible runs.
        """
        params: dict = {"deck": deck, "stake": stake}
        if seed is not None:
            params["seed"] = seed
        return self._call("start", params)

    def menu(self) -> dict:
        """Return to the main menu from any state."""
        return self._call("menu")

    def save(self, path: str) -> dict:
        """Save the current run to a file. Returns {"success": true, "path": ...}."""
        return self._call("save", {"path": path})

    def load(self, path: str) -> dict:
        """Load a saved run from a file. Returns {"success": true, "path": ...}."""
        return self._call("load", {"path": path})

    # -----------------------------------------------------------------------
    # Blind Selection (state=BLIND_SELECT)
    # -----------------------------------------------------------------------

    def select(self) -> dict:
        """Select the current blind to begin the round. Returns gamestate (SELECTING_HAND)."""
        return self._call("select")

    def skip(self) -> dict:
        """Skip the current blind (Small or Big only)."""
        return self._call("skip")

    # -----------------------------------------------------------------------
    # Hand Phase (state=SELECTING_HAND)
    # -----------------------------------------------------------------------

    def play(self, cards: list[int]) -> dict:
        """
        Play cards from hand.

        Args:
            cards: 0-based indices of cards to play (1–5 cards).
        """
        if not 1 <= len(cards) <= 5:
            raise ValueError(f"Must play 1–5 cards, got {len(cards)}")
        return self._call("play", {"cards": cards})

    def discard(self, cards: list[int]) -> dict:
        """
        Discard cards from hand.

        Args:
            cards: 0-based indices of cards to discard.
        """
        if not cards:
            raise ValueError("Must discard at least 1 card")
        return self._call("discard", {"cards": cards})

    def rearrange_hand(self, order: list[int]) -> dict:
        """
        Rearrange cards in hand.

        Args:
            order: New order as a permutation of current indices.
                   e.g. [4, 3, 2, 1, 0] reverses a 5-card hand.
        """
        return self._call("rearrange", {"hand": order})

    # -----------------------------------------------------------------------
    # Round End (state=ROUND_EVAL)
    # -----------------------------------------------------------------------

    def cash_out(self) -> dict:
        """Cash out round rewards and transition to shop (state=SHOP)."""
        return self._call("cash_out")

    # -----------------------------------------------------------------------
    # Shop Phase (state=SHOP)
    # -----------------------------------------------------------------------

    def next_round(self) -> dict:
        """Leave the shop and advance to blind selection (state=BLIND_SELECT)."""
        return self._call("next_round")

    def reroll(self) -> dict:
        """Reroll shop items (costs money)."""
        return self._call("reroll")

    def buy_card(self, index: int) -> dict:
        """Buy a card (Joker or consumable) from the shop by 0-based index."""
        return self._call("buy", {"card": index})

    def buy_voucher(self, index: int) -> dict:
        """Buy a voucher from the shop by 0-based index."""
        return self._call("buy", {"voucher": index})

    def buy_pack(self, index: int) -> dict:
        """Buy a booster pack from the shop by 0-based index."""
        return self._call("buy", {"pack": index})

    def sell_joker(self, index: int) -> dict:
        """Sell a Joker by 0-based index."""
        return self._call("sell", {"joker": index})

    def sell_consumable(self, index: int) -> dict:
        """Sell a consumable by 0-based index."""
        return self._call("sell", {"consumable": index})

    def rearrange_jokers(self, order: list[int]) -> dict:
        """Rearrange Jokers. Order matters for Blueprint/Brainstorm."""
        return self._call("rearrange", {"jokers": order})

    def rearrange_consumables(self, order: list[int]) -> dict:
        """Rearrange consumables."""
        return self._call("rearrange", {"consumables": order})

    # -----------------------------------------------------------------------
    # Booster Pack (state=SMODS_BOOSTER_OPENED)
    # -----------------------------------------------------------------------

    def pack_select(self, card: int, targets: Optional[list[int]] = None) -> dict:
        """
        Select a card from an opened booster pack.

        Args:
            card:    0-based index of card to select from pack.
            targets: For Tarot/Spectral cards that need targets —
                     0-based indices of hand cards to apply the effect to.
        """
        params: dict = {"card": card}
        if targets is not None:
            params["targets"] = targets
        return self._call("pack", params)

    def pack_skip(self) -> dict:
        """Skip selection from an opened booster pack."""
        return self._call("pack", {"skip": True})

    # -----------------------------------------------------------------------
    # Consumable Use (state=SELECTING_HAND or SMODS_BOOSTER_OPENED)
    # -----------------------------------------------------------------------

    def use_consumable(self, index: int, cards: Optional[list[int]] = None) -> dict:
        """
        Use a consumable card.

        Args:
            index: 0-based index of the consumable to use.
            cards: 0-based indices of target hand cards (for consumables
                   that require selection, e.g. The Magician).
        """
        params: dict = {"consumable": index}
        if cards is not None:
            params["cards"] = cards
        return self._call("use", params)

    # -----------------------------------------------------------------------
    # Debug / Testing
    # -----------------------------------------------------------------------

    def add(
        self,
        key: str,
        seal: Optional[str] = None,
        edition: Optional[str] = None,
        enhancement: Optional[str] = None,
        eternal: Optional[bool] = None,
        perishable: Optional[int] = None,
        rental: Optional[bool] = None,
    ) -> dict:
        """
        Add a card to the game (debug/testing).

        Args:
            key:         Card key e.g. "j_joker", "c_fool", "H_A"
            seal:        RED | BLUE | GOLD | PURPLE (playing cards only)
            edition:     FOIL | HOLO | POLYCHROME | NEGATIVE
            enhancement: BONUS | MULT | WILD | GLASS | STEEL | STONE | GOLD | LUCKY
            eternal:     Cannot be sold/destroyed (Jokers only)
            perishable:  Rounds until perish (Jokers only)
            rental:      Costs $1/round (Jokers only)
        """
        params: dict = {"key": key}
        if seal is not None:
            params["seal"] = seal
        if edition is not None:
            params["edition"] = edition
        if enhancement is not None:
            params["enhancement"] = enhancement
        if eternal is not None:
            params["eternal"] = eternal
        if perishable is not None:
            params["perishable"] = perishable
        if rental is not None:
            params["rental"] = rental
        return self._call("add", params)

    def screenshot(self, path: str) -> dict:
        """Take a screenshot. Returns {"success": true, "path": ...}."""
        return self._call("screenshot", {"path": path})

    def set(
        self,
        money: Optional[int] = None,
        chips: Optional[int] = None,
        ante: Optional[int] = None,
        round: Optional[int] = None,
        hands: Optional[int] = None,
        discards: Optional[int] = None,
        shop: Optional[bool] = None,
    ) -> dict:
        """
        Set in-game values (debug/testing). Pass only the values to change.

        Args:
            money:    Set money amount
            chips:    Set chips scored
            ante:     Set ante number
            round:    Set round number
            hands:    Set hands remaining
            discards: Set discards remaining
            shop:     Re-stock shop (SHOP state only)
        """
        params = {
            k: v
            for k, v in {
                "money": money,
                "chips": chips,
                "ante": ante,
                "round": round,
                "hands": hands,
                "discards": discards,
                "shop": shop,
            }.items()
            if v is not None
        }
        if not params:
            raise ValueError("set() requires at least one argument")
        return self._call("set", params)

    # -----------------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------------

    def get_state(self) -> str:
        """Return just the current state string (e.g. 'SELECTING_HAND')."""
        return self.gamestate()["state"]

    def is_alive(self) -> bool:
        """Return True if the API is reachable."""
        try:
            self.health()
            return True
        except Exception:
            return False
