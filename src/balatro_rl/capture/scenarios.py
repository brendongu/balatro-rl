"""Custom gamestate scenario loader.

Defines and loads TOML scenario files that configure a live Balatro game
to a specific state via BalatroClient's ``start``, ``set``, and ``add``
methods. Enables targeted data collection for specific parts of runs
without playing through an entire game.

Scenario TOML format::

    [game]
    deck = "RED"
    stake = "WHITE"
    seed = "EXPERT_001"

    [state]
    ante = 3
    money = 15
    hands = 4
    discards = 3

    [[jokers]]
    key = "j_lusty_joker"

    [[jokers]]
    key = "j_blueprint"
    edition = "FOIL"

    [[consumables]]
    key = "c_magician"

    [[cards]]
    key = "H_A"
    seal = "GOLD"
    edition = "POLYCHROME"

Usage::

    from balatro_rl.capture.scenarios import load_scenario, apply_scenario

    scenario = load_scenario("scenarios/late_game.toml")
    apply_scenario(client, scenario)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from balatro_rl.client import BalatroClient


@dataclass
class CardSpec:
    """Specification for a card to add via the ``add`` API."""
    key: str
    seal: str | None = None
    edition: str | None = None
    enhancement: str | None = None
    eternal: bool | None = None
    perishable: int | None = None
    rental: bool | None = None


@dataclass
class Scenario:
    """Parsed scenario definition."""
    name: str = ""
    description: str = ""
    deck: str = "RED"
    stake: str = "WHITE"
    seed: str | None = None
    ante: int | None = None
    money: int | None = None
    hands: int | None = None
    discards: int | None = None
    jokers: list[CardSpec] = field(default_factory=list)
    consumables: list[CardSpec] = field(default_factory=list)
    cards: list[CardSpec] = field(default_factory=list)
    save_path: str | None = None


def _parse_card_spec(data: dict[str, Any]) -> CardSpec:
    """Parse a card spec from a TOML dict."""
    return CardSpec(
        key=data["key"],
        seal=data.get("seal"),
        edition=data.get("edition"),
        enhancement=data.get("enhancement"),
        eternal=data.get("eternal"),
        perishable=data.get("perishable"),
        rental=data.get("rental"),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a TOML file.

    Args:
        path: Path to the TOML scenario file.

    Returns:
        Parsed Scenario object.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    game = data.get("game", {})
    state = data.get("state", {})
    meta = data.get("meta", {})

    scenario = Scenario(
        name=meta.get("name", Path(path).stem),
        description=meta.get("description", ""),
        deck=game.get("deck", "RED"),
        stake=game.get("stake", "WHITE"),
        seed=game.get("seed"),
        ante=state.get("ante"),
        money=state.get("money"),
        hands=state.get("hands"),
        discards=state.get("discards"),
        jokers=[_parse_card_spec(j) for j in data.get("jokers", [])],
        consumables=[_parse_card_spec(c) for c in data.get("consumables", [])],
        cards=[_parse_card_spec(c) for c in data.get("cards", [])],
        save_path=game.get("save_path"),
    )
    return scenario


def apply_scenario(client: BalatroClient, scenario: Scenario) -> dict[str, Any]:
    """Apply a scenario to a live Balatro game.

    If the scenario specifies a ``save_path``, loads that save file instead
    of starting a new game and applying modifications.

    Args:
        client: Connected BalatroClient instance.
        scenario: Scenario to apply.

    Returns:
        The gamestate after applying the scenario.
    """
    if scenario.save_path:
        client.load(scenario.save_path)
        return client.gamestate()

    # Start a new game
    gs = client.start(
        deck=scenario.deck,
        stake=scenario.stake,
        seed=scenario.seed,
    )

    # Select the first blind to enter SELECTING_HAND (needed for set/add)
    state = gs.get("state", "")
    if state == "BLIND_SELECT":
        gs = client.select()

    # Apply state modifications
    set_kwargs: dict[str, Any] = {}
    if scenario.ante is not None:
        set_kwargs["ante"] = scenario.ante
    if scenario.money is not None:
        set_kwargs["money"] = scenario.money
    if scenario.hands is not None:
        set_kwargs["hands"] = scenario.hands
    if scenario.discards is not None:
        set_kwargs["discards"] = scenario.discards

    if set_kwargs:
        gs = client.set(**set_kwargs)

    # Add jokers
    for spec in scenario.jokers:
        gs = client.add(
            key=spec.key,
            seal=spec.seal,
            edition=spec.edition,
            enhancement=spec.enhancement,
            eternal=spec.eternal,
            perishable=spec.perishable,
            rental=spec.rental,
        )

    # Add consumables
    for spec in scenario.consumables:
        gs = client.add(key=spec.key)

    # Add playing cards
    for spec in scenario.cards:
        gs = client.add(
            key=spec.key,
            seal=spec.seal,
            edition=spec.edition,
            enhancement=spec.enhancement,
        )

    return client.gamestate()
