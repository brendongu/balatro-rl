"""Expert gameplay capture harness.

Two modes for collecting imitation learning data from live Balatro:

- **observer**: passive polling while the expert plays through the game UI
- **interactive**: terminal interface where the expert picks from legal actions

Both support custom gamestate scenario loading via TOML files.
"""
