"""Behavioral cloning training from demonstration data.

Trains a policy network to imitate expert actions using supervised
learning on collected demonstrations. Supports phase filtering to
train phase-specific policies.

Usage::

    uv run python scripts/train_bc.py --data-dir data/demos
    uv run python scripts/train_bc.py --data-dir data/demos --phase 3 --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from balatro_rl.imitation.dataset import DemoDataset


class _TorchDemoDataset(Dataset):
    """Thin torch wrapper around DemoDataset."""

    def __init__(self, demo: DemoDataset) -> None:
        self._demo = demo

    def __len__(self) -> int:
        return len(self._demo)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._demo[idx]
        obs = torch.from_numpy(item["obs_global"])
        mask = torch.from_numpy(item["action_mask"])
        action = torch.tensor(item["action"], dtype=torch.long)
        return obs, mask, action


class BCPolicy(nn.Module):
    """Simple MLP policy for behavioral cloning with action masking."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.net(obs)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


def train_bc(
    data_dir: str,
    phase: int | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 256,
    save_path: str = "models/bc_policy.pt",
    max_episodes: int | None = None,
) -> dict[str, float]:
    demo = DemoDataset(data_dir, phase_filter=phase, max_episodes=max_episodes)
    summary = demo.summary()
    print(f"Dataset: {summary}")

    if len(demo) == 0:
        print("No data found.")
        return {"loss": float("nan"), "accuracy": 0.0}

    obs_dim = demo.obs_global.shape[1]
    n_actions = demo.action_masks.shape[1]

    dataset = _TorchDemoDataset(demo)
    n_val = max(len(dataset) // 10, 1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(obs_dim, n_actions, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for obs, mask, action in train_loader:
            obs, mask, action = obs.to(device), mask.to(device), action.to(device)
            logits = model(obs, mask)
            loss = criterion(logits, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            train_correct += (logits.argmax(dim=1) == action).sum().item()
            train_total += obs.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for obs, mask, action in val_loader:
                obs, mask, action = obs.to(device), mask.to(device), action.to(device)
                logits = model(obs, mask)
                loss = criterion(logits, action)
                val_loss += loss.item() * obs.size(0)
                val_correct += (logits.argmax(dim=1) == action).sum().item()
                val_total += obs.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        print(
            f"  Epoch {epoch + 1:3d}/{epochs} | "
            f"train_loss={train_loss / train_total:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return {"val_loss": best_val_loss, "val_accuracy": val_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavioral cloning policy")
    parser.add_argument("--data-dir", type=str, default="data/demos")
    parser.add_argument("--phase", type=int, default=None,
                        help="Phase filter (0=blind, 1=hand, 3=shop)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--save-path", type=str, default="models/bc_policy.pt")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    phase_names = {0: "BLIND_SELECT", 1: "SELECTING_HAND", 3: "SHOP"}
    phase_str = phase_names.get(args.phase, f"ALL (phase={args.phase})" if args.phase else "ALL")
    print(f"Training BC policy for phase: {phase_str}")

    results = train_bc(
        data_dir=args.data_dir,
        phase=args.phase,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        save_path=args.save_path,
        max_episodes=args.max_episodes,
    )
    print(f"\nFinal: {results}")


if __name__ == "__main__":
    main()
