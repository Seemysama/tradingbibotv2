from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class AttentionLSTM(nn.Module):
    """LSTM avec un bloc d'attention simple pour pondérer les pas de temps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        output, _ = self.lstm(x)  # (batch, seq, hidden)
        weights = torch.softmax(self.attn(output), dim=1)  # (batch, seq, 1)
        context = torch.sum(output * weights, dim=1)  # (batch, hidden)
        return self.head(context)


def train_with_validation(
    X: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    device: torch.device,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    model_path: Path = Path("models/best_model.pth"),
) -> Tuple[AttentionLSTM, float]:
    """Entraîne le modèle et sauvegarde le meilleur sur la loss de validation."""
    dataset = TensorDataset(X, y)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AttentionLSTM(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state: Optional[dict] = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch}/{epochs} Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping déclenché")
                break

    if best_state:
        model.load_state_dict(best_state)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Meilleur modèle sauvegardé: {model_path} (val_loss={best_val:.6f})")
    return model, best_val
